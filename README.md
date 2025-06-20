# Feedback Sentiment Analyzer

This project provides a sentiment analysis system for email feedback using both traditional (VADER) and modern machine learning techniques (DistilBERT).

## Project Structure

- `app/api.py` - FastAPI backend that performs sentiment analysis using VADER and Hugging Face models
- `app/streamlit_app.py` - Streamlit frontend for user interaction
- `requirements.txt` - Required packages for the project

## Setup Instructions

1. **Create and activate a virtual environment (optional but recommended)**

```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python -m venv env
source env/bin/activate
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Start the FastAPI backend**

```bash
# From project root directory
uvicorn app.api:app --reload
```

The API will be available at http://127.0.0.1:8000

- API documentation: http://127.0.0.1:8000/docs
- API endpoints:
  - GET `/` - Welcome message
  - POST `/predict` - Analyze sentiment of provided text

4. **Start the Streamlit frontend**

```bash
# From project root directory (in a new terminal window)
streamlit run app/streamlit_app.py
```

The Streamlit app will be available at http://localhost:8501

## Usage

1. Enter your feedback text in the provided text area
2. Click "Analyze Sentiment" button
3. View the sentiment analysis results from both VADER and DistilBERT models

## How It Works

### VADER Sentiment Analysis
- Rule-based sentiment analyzer specifically attuned to sentiments expressed in social media
- Provides a compound score between -1 (most negative) and 1 (most positive)

### DistilBERT Sentiment Analysis
- A lighter, faster version of BERT, fine-tuned for sentiment analysis on the SST-2 dataset
- Provides a sentiment label (POSITIVE/NEGATIVE) and confidence score

## Requirements

See `requirements.txt` for the full list of dependencies.
