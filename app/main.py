from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Load model & tokenizer
model_path = "./app/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Sentiment tools
hf_sentiment = pipeline("sentiment-analysis")
vader = SentimentIntensityAnalyzer()

# Label map — change if you have different label indexes
label_map = {0: "feature request", 1: "negative", 2: "positive", 3: "technical issue"}

class Feedback(BaseModel):
    text: str

@app.post("/predict/")
def predict(feedback: Feedback):
    # Classification
    inputs = tokenizer(feedback.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = label_map.get(predicted_class, "unknown")

    # Sentiment
    vader_score = vader.polarity_scores(feedback.text)
    hf_result = hf_sentiment(feedback.text)[0]

    return {
        "predicted_label": predicted_label,
        "vader_sentiment": vader_score,
        "hf_sentiment": hf_result
    }
