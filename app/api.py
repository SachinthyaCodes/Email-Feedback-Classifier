from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn

app = FastAPI(title="Feedback Sentiment Classifier API")

# Initialize models
try:
    # Initialize Hugging Face sentiment pipeline
    hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Initialize VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error initializing models: {e}")
    raise

class FeedbackRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    vader_sentiment: str
    vader_score: float
    huggingface_sentiment: str
    huggingface_score: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: FeedbackRequest):
    text = request.text
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Please provide non-empty text")
    
    try:
        # Get VADER sentiment
        vader_scores = vader_analyzer.polarity_scores(text)
        compound_score = vader_scores['compound']
        
        # Determine VADER sentiment label
        if compound_score >= 0.05:
            vader_sentiment = "POSITIVE"
        elif compound_score <= -0.05:
            vader_sentiment = "NEGATIVE"
        else:
            vader_sentiment = "NEUTRAL"
            
        # Get Hugging Face sentiment
        hf_result = hf_sentiment(text)[0]
        hf_sentiment = hf_result['label']
        hf_score = hf_result['score']
        
        return SentimentResponse(
            text=text,
            vader_sentiment=vader_sentiment,
            vader_score=compound_score,
            huggingface_sentiment=hf_sentiment,
            huggingface_score=hf_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sentiment: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Feedback Sentiment Classifier API. Use /predict endpoint to analyze sentiment."}

if __name__ == "__main__":
    # Run the API with uvicorn
    # Will run on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
