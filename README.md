# OpenFold-Test-Martina

Minimal Python CLI for sentiment analysis using Hugging Face `transformers`.

## Project description

This project provides a small command-line interface (CLI) that runs sentiment
analysis on input text using the Hugging Face `pipeline("sentiment-analysis")`
with the default model `distilbert-base-uncased-finetuned-sst-2-english`.

## Environment setup

1. **Create and activate a virtual environment (recommended)**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Run the script

Use the CLI to analyze sentiment of a text snippet:

```bash
python src/hf_inference.py --text "I love this movie"
```

The output will be in the form:

```text
LABEL=<label> SCORE=<score>
```

# view new changes
# new new