from openai import OpenAI
import pandas as pd
import os
import time
from collections import Counter
import numpy as np
import random
import json  # only used if you later switch to JSON mode

from pathlib import Path
import sys

PROMPTS_DIR = Path(__file__).resolve().parent
if not (PROMPTS_DIR / "prompts.py").exists():
    PROMPTS_DIR = PROMPTS_DIR.parent

if str(PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPTS_DIR))

from prompts import DETECTION_SYSTEM_PROMPT, build_detection_user_prompt

# -------------------
# CONFIG
# -------------------
INPUT_FOLDER = "../../data/20_distorted"   # folder of CSVs
OUTPUT_FOLDER = "annotated_results_gpt_4o_mini_2"  # where results go
N_REPEATS = 3                       # how many times to query per cell
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0                    # keep deterministic

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize OpenAI client (reads key from env: OPENAI_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

VALID = {"0", "1"}#, "2"}

SYSTEM_MSG = DETECTION_SYSTEM_PROMPT

#    " - 2 = Benign/generic, no false statement identifiable.\n"

# -------------------
# API classification
# -------------------
def classify_once(text):
    """One API call. Returns '0' | '1' or None on failure/malformed output."""
    user = build_detection_user_prompt(text)
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=1,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip().upper()
        return raw if raw in VALID else None
    except Exception:
        return None

# -------------------
# Main file analysis
# -------------------



def analyze_file(filepath):
    data = pd.read_csv(filepath, keep_default_na=False,sep=";")
    col_order = list(data.columns)
    print(col_order)
    data.drop(columns=['Count', 'Classifcation result', 'Feature Importance'],inplace=True)
    col_order = list(data.columns)
    print(col_order)


    #print(data)

    #print(col_order)

    ####
    # Get predictions
    # Collect majority
    # do average between


    # To store majority votes
    majority_results = pd.DataFrame(index=data.index, columns=col_order)
    # To store all individual predictions
    all_predictions = pd.DataFrame(index=data.index, columns=pd.MultiIndex.from_product([col_order, range(N_REPEATS)],
                                                                                       names=["Column", "Repeat"]))



    for col in col_order:
        for i, cell in enumerate(data[col]):


            votes = []
            for repeat in range(N_REPEATS):
                vote = classify_once(str(cell))
                votes.append(vote)
                all_predictions.at[i, (col, repeat)] = vote
            # Majority vote
            majority_vote = Counter(votes).most_common(1)[0][0]
            majority_results.at[i, col] = majority_vote

    print(majority_results)
    print(all_predictions)


    # Save majority votes
    majority_results = majority_results.apply(pd.to_numeric, errors='coerce')
    base_name = os.path.basename(filepath).replace(".csv", "")
    majority_results.to_csv(os.path.join(OUTPUT_FOLDER, f"{base_name}_majority.csv"), index=False)

    # Save all predictions
    all_predictions.to_csv(os.path.join(OUTPUT_FOLDER, f"{base_name}_all_predictions.csv"))


    # Optional: calculate column-wise agreement
    #column_agreement = {}
    #for col in col_order:
    #    agreements = [Counter(all_predictions.loc[i, col].tolist()).most_common(1)[0][1] / N_REPEATS
    #                  for i in all_predictions.index]
    #    column_agreement[col] = np.mean(agreements)



    #print(f"Column agreement (higher is better): {column_agreement}")

    print(majority_results.mean())



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
