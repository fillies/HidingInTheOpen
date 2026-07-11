# logistic_fit_and_plots_with_prints.py
# Fits a 2-parameter logistic U(d)=expit(k*(d-x0)) to each series,
# saves one plot per series, prints line-by-line results + a table,
# and computes averages across all files.

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import curve_fit
from scipy.special import expit


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
    return x0 + (math.log(tau / (1.0 - tau)) / k) if (k != 0 and 0 < tau < 1) else float('nan')


INPUT_FOLDER = "./annotation_analysis"


def main():
    all_files_summaries = []  # Store all summary tables for averaging

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".csv"):
            continue
        print("##############################################################################")
        print(INPUT_FOLDER)
        print(filename)
        model_name = filename.split("_")[2]

        df = pd.read_csv(os.path.join(INPUT_FOLDER, filename), keep_default_na=False)

        # ---- Fitting, printing, plotting ----
        x = df["distortion_level"].values.astype(float)
        x_dense = np.linspace(x.min() - 0.25, x.max() + 0.25, 400)

        os.makedirs("multimodal_plots_2", exist_ok=True)

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
                print(f"{col}: constant_series | k=NA | x0(MUM@0.5)=NA | R2=NA | adj_R2=NA | RMSE=0.000")
                plt.title(f"{col} — constant series (fit not meaningful)")
                plt.xlabel("distortion_level");
                plt.ylabel(col)
                plt.ylim(-0.05, 1.05);
                plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots_2", f"{col}_{filename}.png")
                plt.tight_layout();
                plt.savefig(out, dpi=200);
                plt.close()
                summary_rows.append({
                    "series": col, "status": "constant_series",
                    "k": np.nan, "x0 (MUM@0.5)": np.nan,
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
                print(
                    f"{col}: status=ok | k={k:.4f} | x0(MUM@0.5)={x0:.4f} | R2={r2:.4f} | adj_R2={adjr2:.4f} | RMSE={rmse:.4f}")

                # Save numbers to table
                summary_rows.append({
                    "series": col, "status": "ok",
                    "k": k, "x0 (MUM@0.5)": x0, "R2": r2, "adj_R2": adjr2, "RMSE": rmse
                })

                # Plot fitted curve + MHUD line
                plt.plot(x_dense, yhat_dense, label="2-param logistic fit")
                plt.axvline(mhud, linestyle="--", alpha=0.8, label=f"MUM@0.5 ≈ {mhud:.3f}")

                plt.ylim(-0.05, 1.05)
                plt.xlabel("distortion_level");
                plt.ylabel(col)
                plt.title(
                    f"{col}_{model_name}\nR²={r2:.3f} | adj R²={adjr2:.3f} | RMSE={rmse:.3f} | k={k:.3f}, x0={x0:.3f}")
                plt.legend(loc="best");
                plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots_2", f"{col}_{filename}.png")
                plt.tight_layout();
                plt.savefig(out, dpi=200);
                plt.close()

            except Exception as e:
                # Print failure line
                print(f"{col}: fit_failed: {e}")
                plt.title(f"{col} — fit failed: {e}")
                plt.xlabel("distortion_level");
                plt.ylabel(col)
                plt.ylim(-0.05, 1.05);
                plt.grid(alpha=0.3)
                out = os.path.join("multimodal_plots_2", f"{col}_{filename}_fit_failed.png")
                plt.tight_layout();
                plt.savefig(out, dpi=200);
                plt.close()

        # ---- Compact table at the end ----
        summary = pd.DataFrame(summary_rows).sort_values("series").reset_index(drop=True)

        # Clamp R2 and adj_R2: set all values < -1 to -1
        summary['R2_clamped'] = summary['R2'].clip(lower=-1)
        summary['adj_R2_clamped'] = summary['adj_R2'].clip(lower=-1)

        summary['filename'] = filename  # Track which file this came from
        all_files_summaries.append(summary)

        # pretty rounding for display
        with pd.option_context('display.max_columns', None, 'display.width', 120):
            print("\n===== Summary Table =====")
            print(summary.round({"k": 4, "x0 (MUM@0.5)": 4, "R2": 4, "adj_R2": 4, "RMSE": 4, "R2_clamped": 4,
                                 "adj_R2_clamped": 4}).to_string(index=False))

        print("\nSaved per-series plots to ./multimodal_plots/")

    # ---- Compute and save averages across all files ----
    if all_files_summaries:
        print("\n\n" + "=" * 80)
        print("COMPUTING AVERAGES ACROSS ALL FILES")
        print("=" * 80)

        combined_df = pd.concat(all_files_summaries, ignore_index=True)

        # Group by series and compute mean (ignoring NaN values)
        avg_summary = combined_df.groupby('series').agg({
            'k': 'mean',
            'x0 (MUM@0.5)': 'mean',
            'R2': 'mean',
            'adj_R2': 'mean',
            'RMSE': 'mean',
            'R2_clamped': 'mean',
            'adj_R2_clamped': 'mean'
        }).reset_index()

        avg_summary = avg_summary.sort_values('series').reset_index(drop=True)

        # Display the averaged table
        with pd.option_context('display.max_columns', None, 'display.width', 120):
            print("\n===== AVERAGED Summary Table (Across All Files) =====")
            print(avg_summary.round({"k": 4, "x0 (MUM@0.5)": 4, "R2": 4, "adj_R2": 4, "RMSE": 4, "R2_clamped": 4,
                                     "adj_R2_clamped": 4}).to_string(index=False))

        # Save to CSV
        output_csv = "averaged_summary_across_files.csv"
        avg_summary.to_csv(output_csv, index=False)
        print(f"\nAveraged summary saved to: {output_csv}")

        # Also save the complete combined data
        combined_output_csv = "all_files_combined_summary.csv"
        combined_df.to_csv(combined_output_csv, index=False)
        print(f"Complete combined summary saved to: {combined_output_csv}")


if __name__ == "__main__":
    main()