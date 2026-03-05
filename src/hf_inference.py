import argparse

from transformers import pipeline


_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def predict_sentiment(text: str) -> dict:
    """
    Run sentiment analysis on the given text and return a dict
    containing the predicted label and score.
    """
    classifier = pipeline("sentiment-analysis", model=_SENTIMENT_MODEL)
    result = classifier(text)[0]
    return {"label": result["label"], "score": float(result["score"])}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Hugging Face sentiment analysis CLI."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prediction = predict_sentiment(args.text)
    label = prediction["label"]
    score = prediction["score"]
    print(f"LABEL={label} SCORE={score}")


if __name__ == "__main__":
    main()
