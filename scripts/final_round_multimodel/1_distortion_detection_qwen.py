import requests
import json
import pandas as pd
import os
import time
from collections import Counter
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
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
OUTPUT_FOLDER = "annotated_results_qwen"  # where results go
N_REPEATS = 3                       # how many times to query per cell
MODEL = "Qwen3-VL-32B-Instruct-FP8"        # OpenRouter model name
TEMPERATURE = 0.0                    # deterministic

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.helmholtz-blablador.fz-juelich.de/v1")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#alias-fast - This alias is for model with a high throughput. As of December 2024, it's Ministral-8B-Instruct-2410.
# Initialize Blablador LLM
# Available models: alias-chat, alias-fast, alias-large, alias-code
# Or specific models like: ministral-8b-instruct-2410, llama-3.1-70b-instruct, etc.
llm = ChatOpenAI(
    model_name="alias-large",  # Use alias-fast for high throughput
    temperature=TEMPERATURE,
    max_tokens=1
)

VALID = {"0", "1"}

SYSTEM_MSG = DETECTION_SYSTEM_PROMPT

# -------------------
# API WRAPPER USING OPENROUTER
# -------------------


# -------------------
# CLASSIFIER
# -------------------
def classify_once(text):
    """One API call using Blablador. Returns '0' | '1' or None on failure."""
    try:
        messages = [
            SystemMessage(content=SYSTEM_MSG),
            HumanMessage(content=build_detection_user_prompt(text))
        ]

        response = llm.invoke(messages)
        print(response)
        raw = response.content.strip().upper()

        return raw if raw in VALID else None

    except Exception as e:
        print(f"Blablador API error: {e}")
        return None
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
                time.sleep(3)

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
