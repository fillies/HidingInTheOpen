# logistic_fit_and_plots_with_prints.py
# Fits a 2-parameter logistic U(d)=expit(k*(d-x0)) to each series,
# saves one plot per series, and prints line-by-line results + a table.

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import curve_fit
from scipy.special import expit

# ---- Paste your CSV here (or replace with pd.read_csv("your_file.csv")) ----
csv_text = """distortion_level,selection_abbreviations,selection_change_in_spelling,selection_change_in_spelling_known,selection_code_phonetic,selection_code_words,selection_emoticons,selection_paraphrasing
0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1,0.0,0.0,0.0,0.0,0.2,0.2,0.0
2,0.2,0.0,0.2,0.4,0.4,0.2,0.2
3,0.6,0.0,0.2,0.4,0.8,0.4,0.2
4,0.8,0.0,0.2,0.6,1.0,0.8,0.2
5,1.0,0.0,0.2,0.8,1.0,1.0,0.6
"""
df = pd.read_csv(StringIO(csv_text))
# df = pd.read_csv("your_file.csv")

# ---- Model & helpers ----
def logistic01(x, k, x0):
    return expit(k * (x - x0))

def fit_2param_logistic(x, y):
    # bounds: k in (1e-6,100], x0 within [min-1, max+1] for stability
    k0, x00 = 1.0, np.median(x)
    bounds = ([1e-6, float(x.min()) - 1.0], [100.0, float(x.max()) + 1.0])
    (k, x0), _ = curve_fit(logistic01, x, y, p0=[k0, x00], bounds=bounds, maxfev=20000)
    return float(k), float(x0)

def metrics(y, yhat, p=2):
    n = len(y)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
    adj = float('nan') if (n <= p + 1 or not np.isfinite(r2)) else 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    rmse = math.sqrt(ss_res / n) if n > 0 else float('nan')
    return r2, adj, rmse

def mhud_tau_2p(k, x0, tau=0.5):
    return x0 + (math.log(tau/(1.0 - tau)) / k) if (k != 0 and 0 < tau < 1) else float('nan')

# ---- Fitting, printing, plotting ----
x = df["distortion_level"].values.astype(float)
x_dense = np.linspace(x.min() - 0.25, x.max() + 0.25, 400)

os.makedirs("plots", exist_ok=True)

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
        print(f"{col}: constant_series | k=NA | x0(MHUD@0.5)=NA | R2=NA | adj_R2=NA | RMSE=0.000")
        plt.title(f"{col} — constant series (fit not meaningful)")
        plt.xlabel("distortion_level"); plt.ylabel(col)
        plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3)
        out = os.path.join("plots", f"{col}.png")
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.show()
        summary_rows.append({
            "series": col, "status": "constant_series",
            "k": np.nan, "x0 (MHUD@0.5)": np.nan,
            "R2": np.nan, "adj_R2": np.nan, "RMSE": 0.0
        })
        continue

    try:
        k, x0 = fit_2param_logistic(x, y)
        yhat = logistic01(x, k, x0)
        yhat_dense = logistic01(x_dense, k, x0)
        r2, adjr2, rmse = metrics(y, yhat, p=2)
        mhud = mhud_tau_2p(k, x0, tau=0.5)

        # Print one concise line for this series
        print(f"{col}: status=ok | k={k:.4f} | x0(MHUD@0.5)={x0:.4f} | R2={r2:.4f} | adj_R2={adjr2:.4f} | RMSE={rmse:.4f}")

        # Save numbers to table
        summary_rows.append({
            "series": col, "status": "ok",
            "k": k, "x0 (MHUD@0.5)": x0, "R2": r2, "adj_R2": adjr2, "RMSE": rmse
        })

        # Plot fitted curve + MHUD line
        plt.plot(x_dense, yhat_dense, label="2-param logistic fit")
        plt.axvline(mhud, linestyle="--", alpha=0.8, label=f"MHUD@0.5 ≈ {mhud:.3f}")

        plt.ylim(-0.05, 1.05)
        plt.xlabel("distortion_level"); plt.ylabel(col)
        plt.title(f"{col}\nR²={r2:.3f} | adj R²={adjr2:.3f} | RMSE={rmse:.3f} | k={k:.3f}, x0={x0:.3f}")
        plt.legend(loc="best"); plt.grid(alpha=0.3)
        out = os.path.join("plots", f"{col}.png")
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.show()

    except Exception as e:
        # Print failure line
        print(f"{col}: fit_failed: {e}")
        plt.title(f"{col} — fit failed: {e}")
        plt.xlabel("distortion_level"); plt.ylabel(col)
        plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3)
        out = os.path.join("plots", f"{col}_fit_failed.png")
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.show()

# ---- Compact table at the end ----
summary = pd.DataFrame(summary_rows).sort_values("series").reset_index(drop=True)
# pretty rounding for display
with pd.option_context('display.max_columns', None, 'display.width', 120):
    print("\n===== Summary Table =====")
    print(summary.round({"k": 4, "x0 (MHUD@0.5)": 4, "R2": 4, "adj_R2": 4, "RMSE": 4}).to_string(index=False))

print("\nSaved per-series plots to ./plots/")
