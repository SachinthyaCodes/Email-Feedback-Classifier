import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import time
import streamlit as st

def analyze_feedback_batch(file_path):
    """Analyzes sentiment of a batch of feedback texts from a CSV file."""
    # Load data
    df = pd.read_csv(file_path)
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")
    
    # Initialize sentiment analyzers
    vader_analyzer = SentimentIntensityAnalyzer()
    hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    results = []
    
    # Process each feedback text
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
            
        # Get Hugging Face sentiment
        hf_result = hf_sentiment(text)[0]
        hf_sentiment_label = hf_result['label']
        hf_score = hf_result['score']
        
        results.append({
            'text': text,
            'vader_sentiment': vader_sentiment,
            'vader_score': compound_score,
            'huggingface_sentiment': hf_sentiment_label,
            'huggingface_score': hf_score
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def get_sentiment_summary(result_df):
    """Generates summary statistics from sentiment analysis results."""
    
    # Count occurrences of each sentiment
    vader_counts = result_df['vader_sentiment'].value_counts()
    hf_counts = result_df['huggingface_sentiment'].value_counts()
    
    # Calculate average scores
    vader_avg_score = result_df['vader_score'].mean()
    hf_avg_score = result_df['huggingface_score'].mean()
    
    # Determine overall sentiment
    overall_sentiment = "POSITIVE" if vader_avg_score > 0 else "NEGATIVE" if vader_avg_score < 0 else "NEUTRAL"
    
    summary = {
        "total_items": len(result_df),
        "vader_sentiment_counts": vader_counts.to_dict(),
        "vader_average_score": vader_avg_score,
        "huggingface_sentiment_counts": hf_counts.to_dict(),
        "huggingface_average_score": hf_avg_score,
        "overall_sentiment": overall_sentiment
    }
    
    return summary

def create_sentiment_visualizations(result_df):
    """Creates visualization charts for sentiment analysis results."""
    
    # Create figure for VADER sentiment
    fig_vader, ax_vader = plt.subplots(figsize=(8, 6))
    vader_counts = result_df['vader_sentiment'].value_counts()
    colors = ['#4CAF50', '#FFC107', '#F44336']
    ax_vader.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    ax_vader.axis('equal')
    
    # Create figure for Hugging Face sentiment
    fig_hf, ax_hf = plt.subplots(figsize=(8, 6))
    hf_counts = result_df['huggingface_sentiment'].value_counts()
    colors = ['#4CAF50', '#F44336']
    ax_hf.pie(hf_counts, labels=hf_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax_hf.axis('equal')
    
    return fig_vader, fig_hf

def get_image_base64(fig):
    """Converts a matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str
