from openai import OpenAI
import pandas as pd
import os
import time
from collections import Counter
import numpy as np
import random
import json
import ast
from difflib import SequenceMatcher

# -------------------
# CONFIG
# -------------------
INPUT_FOLDER = ["./reconstructed_results_claud","./reconstructed_results_gpt_4o","./reconstructed_results_mistral","./reconstructed_results_qwen",
                "./reconstructed_results_xAI","./reconstructed_results_llama","./reconstructed_results_gpt4o_mini"] # folder of CSVs
OUTPUT_FOLDER = "./understanding_analysis"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    start_time = time.time()
    for folder in INPUT_FOLDER:
        aggregated_results = {}

        for filename in os.listdir(folder):
            if not filename.endswith("_understanding.csv"):
                continue
            print("##############################################################################")
            print(folder)
            print(filename)

            data = pd.read_csv(os.path.join(folder, filename), keep_default_na=False)

            data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

            print(data.mean())

            columns_of_interest = ['10_overall_success_rate', '20_overall_success_rate',
                                                             '30_overall_success_rate', '40_overall_success_rate',
                                                             '50_overall_success_rate']
            means = []
            for col in columns_of_interest:
                if col in data.columns:
                    means.append(data[col].mean())
                else:
                    means.append(0)  # If column doesn't exist

            # Use filename (without extension) as the key
            file_key = filename.replace("_majority.csv", "")
            aggregated_results[file_key] = means

            # Create aggregated dataframe
        if aggregated_results:
            # Create dataframe with distortion levels as rows, files as columns
            agg_df = pd.DataFrame(aggregated_results, index=['1', '2',
                                                             '3', '4',
                                                             '5'])

            # Add row 0 at the beginning filled with zeros
            agg_df.loc['0'] = 1
            agg_df = agg_df.sort_index()

            # Add distortion_level as first column
            agg_df.insert(0, 'distortion_level', agg_df.index)
            agg_df = agg_df.reset_index(drop=True)

            # Extract folder name for output file
            folder_name = os.path.basename(folder.rstrip('/\\'))
            output_filename = f"{folder_name}_aggregated.csv"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            agg_df.to_csv(output_path, index=False)
            print(f"\n=== AGGREGATED FILE SAVED ===")
            print(f"Folder: {folder}")
            print(f"Files combined: {len(aggregated_results)}")
            print(f"Saved to: {output_path}")
            print(agg_df)
            print()

    print(f"Total annotation time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
