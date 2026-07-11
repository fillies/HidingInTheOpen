# logistic_fit_and_plots_with_prints_and_logit.py
# Fits a 2-parameter logistic U(d)=expit(k*(d-x0)) to each series,
# also fits a logistic regression logit(p)=β0+β1*d to capture trend,
# saves one plot per series, and prints line-by-line results + a table.

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import curve_fit
from scipy.special import expit
import statsmodels.api as sm

# ---- Model & helpers ----
def logistic01(x, k, x0):
    return expit(k * (x - x0))

def fit_2param_logistic(x, y):
    # bounds: k in (1e-6,100], x0 within [min-1, max+1] for stability
    k0, x00 = 1.0, np.median(x)
    bounds = ([1e-6, float(x.min()) - 1.0], [100.0, float(x.max()) + 1.0])
    (k, x0), _ = curve_fit(logistic01, x, y, p0=[k0, x00],
                           bounds=bounds, maxfev=20000)
    return float(k), float(x0)

def metrics(y, yhat, p=2):
    n = len(y)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
    adj = float('nan') if (n <= p + 1 or not np.isfinite(r2)) else \
        1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    rmse = math.sqrt(ss_res / n) if n > 0 else float('nan')
    return r2, adj, rmse

def mhud_tau_2p(k, x0, tau=0.5):
    return x0 + (math.log(tau/(1.0 - tau)) / k) if (k != 0 and 0 < tau < 1) else float('nan')

# NEW: logistic regression helper for general trend
def logistic_reg_trend(x, y):
    """
    Fit logit(p) = β0 + β1 * x using GLM Binomial.
    Here y is a proportion in [0,1]. We treat each row as 1 'trial';
    this is fine for estimating the sign of β1.
    """
    eps = 1e-6
    y_clip = np.clip(y, eps, 1 - eps)  # avoid exactly 0 or 1
    X = sm.add_constant(x.astype(float))
    model = sm.GLM(y_clip, X, family=sm.families.Binomial())
    res = model.fit()
    beta0, beta1 = res.params
    pval = res.pvalues[1]  # p-value for slope
    return beta0, beta1, pval, res

INPUT_FOLDER = "./annotation_analysis"

def main():
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".csv"):
            continue
        print("##############################################################################")
        print(INPUT_FOLDER)
        print(filename)
        model_name = filename.split("_")[2]

        df = pd.read_csv(os.path.join(INPUT_FOLDER, filename), keep_default_na=False)

        x = df["distortion_level"].values.astype(float)
        x_dense = np.linspace(x.min() - 0.25, x.max() + 0.25, 400)

        os.makedirs("multimodal_plots", exist_ok=True)

        summary_rows = []

        for col in df.columns:
            if col == "distortion_level":
                continue

            y = df[col].values.astype(float)

            # Plot base
            plt.figure(figsize=(6.5, 4.2))
            plt.scatter(x, y, label="Observed", zorder=3)

            if np.allclose(y, y[0]):
                # Constant series: print & plot note
                print(f"{col}: constant_series | k=NA | x0(MUM@0.5)=NA | "
                      f"R2=NA | adj_R2=NA | RMSE=0.000 | "
                      f"logit_beta1=NA | logit_p=NA")
                plt.title(f"{col} — constant series (fit not meaningful)")
                plt.xlabel("distortion_level"); plt.ylabel(col)
                plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots", f"{col}_{filename}.png")
                plt.tight_layout(); plt.savefig(out, dpi=200); #plt.show()
                summary_rows.append({
                    "series": col,
                    "status": "constant_series",
                    "k": np.nan,
                    "x0 (MUM@0.5)": np.nan,
                    "R2": np.nan,
                    "adj_R2": np.nan,
                    "RMSE": 0.0,
                    "beta0_logit": np.nan,
                    "beta1_logit": np.nan,
                    "p_beta1": np.nan
                })
                continue

            try:
                # 1) Logistic regression for general trend
                beta0, beta1, pval, glm_res = logistic_reg_trend(x, y)
                direction = "decrease" if beta1 < 0 else "increase"
                # you can tighten this if you want, e.g. require p<0.05

                # 2) 2-parameter logistic curve fit (your original)
                k, x0 = fit_2param_logistic(x, y)
                yhat = logistic01(x, k, x0)
                yhat_dense = logistic01(x_dense, k, x0)
                r2, adjr2, rmse = metrics(y, yhat, p=2)
                mhud = mhud_tau_2p(k, x0, tau=0.5)

                # Print concise line for this series
                print(f"{col}: status=ok | k={k:.4f} | x0(MUM@0.5)={x0:.4f} | "
                      f"R2={r2:.4f} | adj_R2={adjr2:.4f} | RMSE={rmse:.4f}")
                print(f"   logistic_reg: beta0={beta0:.4f} | beta1={beta1:.4f} "
                      f"({direction}) | p_beta1={pval:.4g}")

                # Save numbers to table
                summary_rows.append({
                    "series": col,
                    "status": "ok",
                    "k": k,
                    "x0 (MUM@0.5)": x0,
                    "R2": r2,
                    "adj_R2": adjr2,
                    "RMSE": rmse,
                    "beta0_logit": beta0,
                    "beta1_logit": beta1,
                    "p_beta1": pval
                })

                # 3) Plot fitted curves
                plt.plot(x_dense, yhat_dense, label="2-param logistic fit")

                # Optional: overlay logistic regression trend as a dashed curve
                yhat_logit_dense = expit(beta0 + beta1 * x_dense)
                plt.plot(x_dense, yhat_logit_dense,
                         linestyle="--", alpha=0.8,
                         label="logistic regression trend")

                # Mark MUM@0.5 from 2-param model
                plt.axvline(mhud, linestyle=":", alpha=0.8,
                            label=f"MUM@0.5 ≈ {mhud:.3f}")

                plt.ylim(-0.05, 1.05)
                plt.xlabel("distortion_level"); plt.ylabel(col)
                plt.title(
                    f"{col}_{model_name}\n"
                    f"R²={r2:.3f} | adj R²={adjr2:.3f} | RMSE={rmse:.3f} | "
                    f"k={k:.3f}, x0={x0:.3f} | "
                    f"β1(logit)={beta1:.3f}, p={pval:.3g}"
                )
                plt.legend(loc="best"); plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots", f"{col}_{filename}.png")
                plt.tight_layout(); plt.savefig(out, dpi=200); #plt.show()

            except Exception as e:
                print(f"{col}: fit_failed: {e}")
                plt.title(f"{col} — fit failed: {e}")
                plt.xlabel("distortion_level"); plt.ylabel(col)
                plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots",
                                   f"{col}_{filename}_fit_failed.png")
                plt.tight_layout(); plt.savefig(out, dpi=200); #plt.show()

                summary_rows.append({
                    "series": col,
                    "status": f"fit_failed: {e}",
                    "k": np.nan,
                    "x0 (MUM@0.5)": np.nan,
                    "R2": np.nan,
                    "adj_R2": np.nan,
                    "RMSE": np.nan,
                    "beta0_logit": np.nan,
                    "beta1_logit": np.nan,
                    "p_beta1": np.nan
                })

        # ---- Compact table at the end ----
        summary = pd.DataFrame(summary_rows).sort_values("series").reset_index(drop=True)

        with pd.option_context('display.max_columns', None, 'display.width', 160):
            print("\n===== Summary Table =====")
            print(
                summary.round({
                    "k": 4,
                    "x0 (MUM@0.5)": 4,
                    "R2": 4,
                    "adj_R2": 4,
                    "RMSE": 4,
                    "beta0_logit": 4,
                    "beta1_logit": 4,
                    "p_beta1": 4
                }).to_string(index=False)
            )

        print("\nSaved per-series plots to ./multimodal_plots/")

if __name__ == "__main__":
    main()
