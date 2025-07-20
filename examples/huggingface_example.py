from transformers import pipeline


def run_sentiment_analysis(text: str = "I love using Transformers models!"):
    """Run sentiment analysis using a pretrained Hugging Face model."""
    classifier = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(text)[0]
    print(f"label: {result['label']}, score: {result['score']:.2f}")


if __name__ == "__main__":
    run_sentiment_analysis()
