from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn
import pandas as pd
from io import StringIO
from typing import List, Dict, Any
import csv

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
    
class BatchSentimentResult(BaseModel):
    text: str
    vader_sentiment: str
    vader_score: float
    huggingface_sentiment: str
    huggingface_score: float
    
class BatchSentimentResponse(BaseModel):
    results: List[BatchSentimentResult]
    summary: Dict[str, Any]

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
        hf_sentiment_label = hf_result['label']
        hf_score = hf_result['score']
        
        return SentimentResponse(
            text=text,
            vader_sentiment=vader_sentiment,
            vader_score=compound_score,
            huggingface_sentiment=hf_sentiment_label,
            huggingface_score=hf_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sentiment: {str(e)}")

@app.post("/batch_predict", response_model=BatchSentimentResponse)
async def batch_predict_sentiment(file: UploadFile = File(...)):
    """
    Process a CSV file with feedback texts and return sentiment analysis for each row plus summary statistics.
    The CSV should have a column named 'text' containing the feedback texts.
    """
    # Check if file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read the CSV file
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # Check if the file is empty
        if not content_str.strip():
            raise HTTPException(status_code=400, detail="The CSV file is empty")
        
        # Use StringIO to create a file-like object for pandas
        df = pd.read_csv(StringIO(content_str))
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain a 'text' column")
        
        # Filter out empty texts
        df = df[df['text'].notna() & (df['text'] != '')]
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid text entries found in CSV")
        
        # Process each text
        results = []
        
        for text in df['text']:
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
                
            # Get Hugging Face sentiment - batch processing would be more efficient here
            hf_result = hf_sentiment(text)[0]
            hf_sentiment_label = hf_result['label']
            hf_score = hf_result['score']
            
            results.append(BatchSentimentResult(
                text=text,
                vader_sentiment=vader_sentiment,
                vader_score=compound_score,
                huggingface_sentiment=hf_sentiment_label,
                huggingface_score=hf_score
            ))
        
        # Calculate summary statistics
        vader_sentiments = [r.vader_sentiment for r in results]
        hf_sentiments = [r.huggingface_sentiment for r in results]
        
        # Count occurrences of each sentiment
        vader_counts = {
            "POSITIVE": vader_sentiments.count("POSITIVE"),
            "NEUTRAL": vader_sentiments.count("NEUTRAL"),
            "NEGATIVE": vader_sentiments.count("NEGATIVE")
        }
        
        hf_counts = {
            "POSITIVE": hf_sentiments.count("POSITIVE"),
            "NEGATIVE": hf_sentiments.count("NEGATIVE")
        }
        
        # Calculate average scores
        vader_avg_score = sum(r.vader_score for r in results) / len(results)
        hf_avg_score = sum(r.huggingface_score for r in results) / len(results)
        
        summary = {
            "total_items": len(results),
            "vader_sentiment_counts": vader_counts,
            "vader_average_score": vader_avg_score,
            "huggingface_sentiment_counts": hf_counts,
            "huggingface_average_score": hf_avg_score,
            "overall_sentiment": "POSITIVE" if vader_avg_score > 0 else "NEGATIVE" if vader_avg_score < 0 else "NEUTRAL"
        }
        
        return BatchSentimentResponse(results=results, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch sentiment: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Feedback Sentiment Classifier API. Use /predict endpoint to analyze sentiment or /batch_predict to analyze a CSV file."}

if __name__ == "__main__":
    # Run the API with uvicorn
    # Will run on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
