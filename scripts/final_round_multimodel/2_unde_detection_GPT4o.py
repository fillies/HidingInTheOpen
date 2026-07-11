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

from pathlib import Path
import sys

PROMPTS_DIR = Path(__file__).resolve().parent
if not (PROMPTS_DIR / "prompts.py").exists():
    PROMPTS_DIR = PROMPTS_DIR.parent

if str(PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTS_DIR))

from prompts import UNDERSTANDING_SYSTEM_PROMPT, build_reconstruction_user_prompt

# -------------------
# CONFIG
# -------------------
INPUT_FOLDER = "../../data/20_understanding"   # folder of CSVs
OUTPUT_FOLDER = "reconstructed_results_gpt_4o"  # where results go
N_REPEATS = 1                       # how many times to query per cell
MODEL = "gpt-4o"
TEMPERATURE = 0.0                    # keep deterministic

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize OpenAI client (reads key from env: OPENAI_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

VALID = {"0", "1"}#, "2"}

SYSTEM_MSG = UNDERSTANDING_SYSTEM_PROMPT

#    " - 2 = Benign/generic, no false statement identifiable.\n"

# -------------------
# API classification
# -------------------
def reconstruct_once(text, words_modulated):

    """One API call. Returns the reconstructed sentence or None on failure."""
    user = build_reconstruction_user_prompt(text, words_modulated)
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=100,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception:
        return None

# -------------------
# Main file analysis
# -------------------

def check_similarity(original:list, predicted:list) -> (int, list):
    similar_flags = []
    for index, word in enumerate(predicted):
        sim = similarity_ratio(original[index].lower(), word)
        if sim >= 0.95:  # within 5% difference
            similar_flags.append(1)
        else:
            similar_flags.append(0)

    return (sum(similar_flags)/len(similar_flags)),similar_flags





def similarity_ratio(a, b):
    """Return a similarity ratio between 0 and 1."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def analyze_file(filepath):
    # Load the data
    data = pd.read_csv(filepath, keep_default_na=False, sep=";")
    data.drop(columns=['Count', 'Classifcation result', 'Feature Importance'], inplace=True, errors='ignore')

    #print(data)
    #print(data.columns)
    #print(data['orignal_terms'])
    #print(data['modulated'])
    if "text" not in data.columns:
        raise ValueError("Expected a 'text' column in the CSV for comparison.")

    # Identify all distorted columns (everything except 'text')
    #distorted_cols = [c for c in data.columns if c != "text" and ]

    texts = data['text']
    #10;20;30;40;50;modulated_terms;original_terms;orignal_terms;modulated
    distorted_cols = data[['10','20','30','40','50']]
    #original_terms = data['orignal_terms']
    #modulated = data['modulated']


    #for index, text in enumerate(texts):

    results_rows = []

    #for index, row in data.iterrows():
    #    print(row)
    #    print(index)
    results = pd.DataFrame(index=data.index)
    results["text"] = data["text"]
    results["orignal_terms"] = data["orignal_terms"]


    #results_prediction_all_columns = []
    #result_success_rate_all_columns = []
    #result_similar_flags_all_columns = []
    distorted_columns = ['10','20','30','40','50'] #'10',
    for index_column, distorted_column in enumerate(distorted_columns):
        sentences = data[distorted_column]
        results_prediction_one_column = []
        result_success_rate_one_columns = []
        result_similar_flags_one_columns = []
        result_words_original = []
        for index_row, sentence in enumerate(sentences):

            #print(data['orignal_terms'][index_row])
            #print(data['modulated'][index_row])
            #print(type(data['modulated'][index_row]))
            #modul = data['modulated'][index_row]

            words_modulated = [word.strip() for word in data['modulated'][index_row].strip('[]').replace('"','').replace("'","").split(',')][:index_column + 1]
            words_original = [word.strip() for word in data['orignal_terms'][index_row].strip('[]').replace('"','').replace("'","").split(',')][:index_column + 1]

            #words_original = ast.literal_eval(data['orignal_terms'][index_row])[:index_column + 2]
            #words_modulated = ast.literal_eval(data['modulated'][index_row])[:index_column + 2]

            predictions = ast.literal_eval(reconstruct_once(sentence, words_modulated).lower().replace("n't",'n’t'))[:index_column + 1]

            results_prediction_one_column.append(predictions)

            print(sentence)
            print('words_original')
            print(words_original)
            print('words_modulated')
            print(words_modulated)

            print('predictions')
            print(predictions)

            overall_success_rate, similar_flags = check_similarity(words_original,predictions)

            #print('overall_success_rate')
            #print(overall_success_rate)
            #print('similar_flags')
            #print(similar_flags)

            result_success_rate_one_columns.append(overall_success_rate)
            result_similar_flags_one_columns.append(similar_flags)
            result_words_original.append(words_original)


        #results_prediction_all_columns.append(results_prediction_one_column)
        #result_success_rate_all_columns.append(result_success_rate_one_columns)
        #result_similar_flags_all_columns.append(result_similar_flags_one_columns)

        print("Done round: " + str(distorted_column))

        results[f"{distorted_column}words_original"] = result_words_original
        results[f"{distorted_column}_predictions"] = results_prediction_one_column
        results[f"{distorted_column}_similar_flags"] = result_similar_flags_one_columns
        results[f"{distorted_column}_overall_success_rate"] = result_success_rate_one_columns

    base_name = os.path.basename(filepath).replace(".csv", "")
    out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_understanding.csv")
    results.to_csv(out_path, index=False)
    print(f"✅ Saved reconstructed file: {out_path}")


    sum_10 = sum(results['10_overall_success_rate'])/20
    print('10_overall_success_rate '+ str(sum_10))
    sum_20 = sum(results['20_overall_success_rate'])/20
    print('20_overall_success_rate '+ str(sum_20))
    sum_30 = sum(results['30_overall_success_rate'])/20
    print('30_overall_success_rate '+ str(sum_30))
    sum_40 = sum(results['40_overall_success_rate'])/20
    print('40_overall_success_rate '+ str(sum_40))
    sum_50 = sum(results['50_overall_success_rate'])/20
    print('50_overall_success_rate'+ str(sum_50))

    print('###############################################################################')

    '''# Create result structure
    results = pd.DataFrame(index=data.index)

    for col in distorted_cols:
        print(f"Processing column: {col}")
        reconstructed_col = []
        similarity_col = []
        similar_flag_col = []




        for i, row in data.iterrows():
            distorted = str(row[col]).strip()
            reference = str(row["text"]).strip()

            reconstructed = reconstruct_once(distorted)
            sim = similarity_ratio(reference, reconstructed)
            is_similar = sim >= 0.95  # within 5% difference

            reconstructed_col.append(reconstructed)
            similarity_col.append(round(sim, 3))
            similar_flag_col.append(is_similar)

            print(f"[{col}] → {distorted} → {reconstructed} | Sim: {sim:.3f} | Similar: {is_similar}")

        # Add new columns for each distorted version
        results[f"{col}_Reconstructed"] = reconstructed_col
        results[f"{col}_Similarity"] = similarity_col
        results[f"{col}_Is_Similar"] = similar_flag_col

    # Keep the reference text for clarity
    results["Reference_text"] = data["text"]

    # Save output
    base_name = os.path.basename(filepath).replace(".csv", "")
    out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_understanding.csv")
    results.to_csv(out_path, index=False)
    print(f"✅ Saved reconstructed file: {out_path}")'''


def main():
    start_time = time.time()
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".csv"):
            continue
        print("##############################################################################")
        print(filename)
        analyze_file(os.path.join(INPUT_FOLDER, filename))
    print(f"Total annotation time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
