import gradio as gr

from src.hf_inference import predict_sentiment


def analyze_sentiment(text: str):
    if not text or not text.strip():
        return "No sentiment yet.", "", ""

    prediction = predict_sentiment(text)
    label = prediction["label"]
    score = prediction["score"]
    score_str = f"{score:.4f}"

    if label.upper() == "POSITIVE":
        message = "This text sounds positive and upbeat."
    else:
        message = "This text expresses negative or cautious sentiment."

    return f"Sentiment: **{label}**", f"Score: **{score_str}**", message


def clear_fields():
    return "", "", "", ""


def build_blocks_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Sentiment Analyzer

            Analyze the sentiment of your text using a Hugging Face Transformers model
            (`distilbert-base-uncased-finetuned-sst-2-english`).
            """
        )

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="Type a sentence to analyze sentiment...",
                    lines=4,
                )
                analyze_btn = gr.Button("Analyze")
                clear_btn = gr.Button("Clear")

                gr.Examples(
                    examples=[
                        ["I love this movie, it's fantastic!"],
                        ["This is the worst experience I've ever had."],
                        ["The product is okay, nothing special."],
                    ],
                    inputs=text_input,
                    label="Try an example",
                )

            with gr.Column():
                result_label = gr.Markdown("Sentiment: **N/A**")
                result_score = gr.Markdown("Score: **N/A**")
                result_message = gr.Markdown("Enter some text to see the analysis here.")

        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=text_input,
            outputs=[result_label, result_score, result_message],
        )

        clear_btn.click(
            fn=clear_fields,
            inputs=None,
            outputs=[text_input, result_label, result_score, result_message],
        )

    return demo


def build_interface_demo():
    return gr.Interface(
        fn=analyze_sentiment,
        inputs=gr.Textbox(
            label="Input text",
            placeholder="Type a sentence to analyze sentiment...",
            lines=4,
        ),
        outputs=[
            gr.Markdown(label="Sentiment"),
            gr.Markdown(label="Score"),
            gr.Markdown(label="Message"),
        ],
        examples=[
            ["I love this movie, it's fantastic!"],
            ["This is the worst experience I've ever had."],
            ["The product is okay, nothing special."],
        ],
        title="Sentiment Analyzer",
        description=(
            "Analyze the sentiment of your text using a Hugging Face Transformers "
            "model (`distilbert-base-uncased-finetuned-sst-2-english`)."
        ),
    )


def build_demo():
    if hasattr(gr, "Blocks"):
        return build_blocks_demo()
    return build_interface_demo()


if __name__ == "__main__":
    app = build_demo()
    app.launch()

