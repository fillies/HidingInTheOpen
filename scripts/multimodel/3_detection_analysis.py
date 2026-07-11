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
#INPUT_FOLDERS = ["./annotated_results_claud","./annotated_results_gpt_4o","./annotated_results_mistral","./annotated_results_qwen", "./annotated_results_xAI","./annotated_results_llama"] # folder of CSVs
INPUT_FOLDERS = ["./annotated_results_gpt4o_mini"]
OUTPUT_FOLDER = "./annotation_analysis"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    start_time = time.time()
    for folder in INPUT_FOLDERS:
        aggregated_results = {}

        for filename in os.listdir(folder):
            if not filename.endswith("_majority.csv"):
                continue
            print("##############################################################################")
            print(folder)
            print(filename)

            data = pd.read_csv(os.path.join(folder, filename), keep_default_na=False)

            data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

            print(data.mean())

            columns_of_interest = ['10', '20', '30', '40', '50']
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
            agg_df = pd.DataFrame(aggregated_results, index=['1', '2', '3', '4', '5'])

            # Add row 0 at the beginning filled with zeros
            agg_df.loc['0'] = 0
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
