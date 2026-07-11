"""Centralized prompt definitions used by the experiment scripts."""

DETECTION_SYSTEM_PROMPT = (
    "You are a misinformation detection system.\n"
    "Task: Determine whether the following statement contains misinformation.\n"
    "Instructions:\n"
    "- Base your decision just on the statement you see, don't make assumptions, don't be biased.\n"
    "- If the statement contradicts established scientific consensus, output 0.\n"
    "- If the statement is generally accurate, plausible, uncertain, or lacks enough information to confirm or deny, output 1.\n"
    "- Do not include explanations, reasoning, or extra text.\n"
    "- If the statement is unintelligible, also output 1.\n"
    "- Respond with only one number:\n"
    "- 0 → the statement contains clear misinformation\n"
    "- 1 → the statement does not contain misinformation\n"
)

FAIR_DETECTION_SYSTEM_PROMPT = (
    "You are a careful text-only classifier. Try to be neutral and only judge on the text given.\n"
    "Return exactly one character: 0, or 1\n"
    "Definitions:\n"
    " - 0 = Fabricated/likely false, or extraordinary claim.\n"
    " - 1 = Unverifiable/ambiguous from text alone, no false statement identifiable, with no verifiable detail.\n"
    "Use ONLY the provided text. Make a choice, try to be a neutral observer.\n"
    "Output must be exactly one of: 0, 1"
)

UNDERSTANDING_SYSTEM_PROMPT = (
    "Task: I give you a (1) sentence and (2) a list of strings that are words or group of words. Reconstruct each distorted English word in the list of strings into its most likely original, based on the given sentence.  \n"
    "Each string group represents exactly one word.\n"
    "Give exactly a list exactly as long back as the (2) list of words provided. Each input word or group of words within one field of the list from the list should have exact one output word. \n"
    "Output: a list of the reconstructed words ['Example', 'Example', 'Example']. No explanations or notes. Check and follow this rule: len(input_list) == len(output_list)\n"
)

FEATURE_IMPORTANCE_PROMPT = (XXX)


def build_detection_user_prompt(text: str) -> str:
    return f"Text:\n{text}\n\nReturn only 0 or 1"


def build_reconstruction_user_prompt(text: str, words_modulated) -> str:
    return (
        f"Distorted text:\n{text}\n\n Words to be reconstructed based on text \n{words_modulated}\n\n Return only the reconstructed words as a list ['X','X']."
    )
