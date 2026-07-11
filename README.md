# Hiding in the Open

Code and data for the research paper "Algospeak, Hiding in the Open: The Trade-off Between Legible Meaning and  Detection Avoidance". Containing experiments on misinformation detection and text reconstruction under controlled text distortions.

The feature-importance identification, incremental-modification procedure that and the 700 populated dataset items remain restricted to vetted researchers after publication.

## What is in this repository

- `scripts/`: Core scripts for annotation, model runs, and analysis.
- `scripts/multimodel/`: Multi-model experiment scripts and post-processing.
- Prompts 


## What is not in this repository

- Data
- feature-importance identification

After publication these can be requested by validated researchers from XXX.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local env file and set keys:

```bash
cp .env.example .env
```

Required variables depend on which scripts you run:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENAI_API_BASE` (optional; defaults to Blablador endpoint in relevant scripts)

## Running experiments

Examples from the repository root:

```bash
python scripts/annotate_fake_easy_eval.py
python scripts/multimodel/1_distortion_detection_GPT4.py
python scripts/multimodel/2_unde_detection_GPT4o.py
python scripts/multimodel/3_detection_analysis.py
```

Most scripts expect input CSV files under `data/` and will write outputs into local result folders.

## Notes for reproducibility

- Keep API keys only in environment variables or local `.env` files.
- Do not commit generated outputs unless you need fixed artifacts for reporting.
- If you prepare a camera-ready release, pin exact package versions in `requirements.txt`.

## Reproducibility and access policy

The feature-importance identification and incremental-modification procedure—that is, the method for finding the words most responsible for a misinformation classification and replacing them to construct high-evasion variants—is treated as an operational construction recipe rather than an evaluation artifact. Consistent with the existing ethics framing, this procedure and the 700 populated dataset items remain restricted to vetted researchers after publication.

## Citation

If this repository accompanies your paper, add the bibliographic entry here (BibTeX recommended).
