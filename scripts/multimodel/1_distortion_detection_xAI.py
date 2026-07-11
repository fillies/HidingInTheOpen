import requests
import json
import pandas as pd
import os
import time
from collections import Counter
import numpy as np
import random

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
OUTPUT_FOLDER = "annotated_results_xAI"  # where results go
N_REPEATS = 3                       # how many times to query per cell
MODEL = "x-ai/grok-4.1-fast:free"        # OpenRouter model name
TEMPERATURE = 0.0                    # deterministic

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("Missing OPENROUTER_API_KEY environment variable.")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VALID = {"0", "1"}

SYSTEM_MSG = DETECTION_SYSTEM_PROMPT

# -------------------
# API WRAPPER USING OPENROUTER
# -------------------
def openrouter_chat(messages, model=MODEL, temperature=TEMPERATURE):
    """
    Sends messages to OpenRouter API and returns the raw completion text.
    """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # Optional but recommended to avoid throttling:
                "HTTP-Referer": "http://localhost",
                "X-Title": "Batch-Misinformation-Classifer",
            },
            data=json.dumps({
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1
            }),
            timeout=40,
        )

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("OpenRouter API error:", e)
        return None

# -------------------
# CLASSIFIER
# -------------------
def classify_once(text):
    """One API call. Returns '0' | '1' or None on failure."""
    user_msg = build_detection_user_prompt(text)

    raw = openrouter_chat(
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ]
    )

    if raw is None:
        return None

    raw = raw.strip().upper()
    return raw if raw in VALID else None

# -------------------
# PROCESS A SINGLE CSV FILE
# -------------------
def analyze_file(filepath):
    data = pd.read_csv(filepath, keep_default_na=False, sep=";")
    col_order = list(data.columns)
    print(col_order)

    # Drop unnecessary columns
    data.drop(columns=['Count', 'Classifcation result', 'Feature Importance'], inplace=True)
    col_order = list(data.columns)
    print(col_order)

    # Prepare output DF structures
    majority_results = pd.DataFrame(index=data.index, columns=col_order)
    all_predictions = pd.DataFrame(
        index=data.index,
        columns=pd.MultiIndex.from_product([col_order, range(N_REPEATS)], names=["Column", "Repeat"])
    )

    # Loop over each cell
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

    # Save outputs
    majority_results = majority_results.apply(pd.to_numeric, errors='coerce')
    base_name = os.path.basename(filepath).replace(".csv", "")
    majority_results.to_csv(os.path.join(OUTPUT_FOLDER, f"{base_name}_majority.csv"), index=False)
    all_predictions.to_csv(os.path.join(OUTPUT_FOLDER, f"{base_name}_all_predictions.csv"))

    print(majority_results.mean())

# -------------------
# MAIN
# -------------------
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
