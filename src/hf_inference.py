import argparse
from functools import lru_cache

from transformers import pipeline


_DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@lru_cache(maxsize=4)
def _get_classifier(model_name: str = _DEFAULT_MODEL):
    return pipeline("sentiment-analysis", model=model_name)


def predict_sentiment(
    text: str, model_name: str = _DEFAULT_MODEL
) -> dict:
    """
    Run sentiment analysis on the given text and return a dict
    containing the predicted label and score.
    """
    classifier = _get_classifier(model_name)
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
    print(f"Sentiment={label} Confidence={score}")


if __name__ == "__main__":
    main()
