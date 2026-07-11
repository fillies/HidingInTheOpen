import requests
import json
import pandas as pd
import os
import time
from collections import Counter
import numpy as np
import random
import anthropic

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
OUTPUT_FOLDER = "annotated_results_claud"  # where results go
N_REPEATS = 3                       # how many times to query per cell
MODEL = "claude-sonnet-4-5"        # OpenRouter model name
TEMPERATURE = 0.0                    # deterministic

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing ANTHROPIC_API_KEY environment variable.")

client = anthropic.Anthropic(api_key = API_KEY )


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VALID = {"0", "1"}

SYSTEM_MSG = CLAUDE_DETECTION_SYSTEM_PROMPT

# -------------------
# API WRAPPER USING OPENROUTER
# -------------------
def openrouter_chat(give_messages, model=MODEL, temperature=TEMPERATURE):
    """
    Sends messages to OpenRouter API and returns the raw completion text.
    """
    #try:
    message = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=give_messages
    )
    #print(message)
    if len(message.content) == 0:
        print(message)
        if message.stop_reason == 'refusal':
            return "0"
        else:
            return '0'
    else:
        return message.content[0].text.strip()

    #except Exception as e:
    #    print("OpenRouter API error:", e)
     #   return None

# -------------------
# CLASSIFIER
# -------------------
def classify_once(text):
    """One API call. Returns '0' | '1' or None on failure."""
    user_msg = build_detection_user_prompt(text)

    raw = openrouter_chat(
        give_messages=[
            {"role": "assistant", "content": SYSTEM_MSG},
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
