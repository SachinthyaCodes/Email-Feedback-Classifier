import streamlit as st
import requests
import json
import pandas as pd

# Define API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Page config
st.set_page_config(
    page_title="Feedback Sentiment Analyzer",
    page_icon="📧",
    layout="centered"
)

# Title and description
st.title("📧 Feedback Sentiment Analyzer")
st.markdown("""
This application analyzes the sentiment of customer feedback using two different methods:
- **VADER** - Rule-based sentiment analyzer specifically attuned to sentiments expressed in social media
- **DistilBERT** - A lighter, faster version of BERT, fine-tuned for sentiment analysis
""")

# Text input
feedback_text = st.text_area("Enter feedback text to analyze", height=150)

# Function to classify sentiment
def get_sentiment(text):
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": text})
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Error: Cannot connect to API. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# Analyze button with a unique key
analyze_button = st.button("Analyze Sentiment", key="analyze_btn")

# Check if the button was clicked
if analyze_button:
    if not feedback_text or feedback_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = get_sentiment(feedback_text)
            
            if result:
                st.success("✅ Sentiment analysis complete!")
                
                # Create two columns for the results
                col1, col2 = st.columns(2)
                  # VADER results
                with col1:
                    st.subheader("VADER Analysis")
                    vader_scores = result["vader_sentiment"]
                    compound_score = vader_scores["compound"]
                    
                    # Determine sentiment label
                    if compound_score >= 0.05:
                        vader_sentiment = "POSITIVE"
                        score_color = "green"
                    elif compound_score <= -0.05:
                        vader_sentiment = "NEGATIVE"
                        score_color = "red"
                    else:
                        vader_sentiment = "NEUTRAL"
                        score_color = "gray"
                    
                    st.markdown(f"**Sentiment:** <span style='color:{score_color};font-weight:bold'>{vader_sentiment}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Compound Score:** <span style='color:{score_color};font-weight:bold'>{compound_score:.4f}</span>", unsafe_allow_html=True)
                    
                    # Show detailed scores
                    st.caption(f"Positive: {vader_scores['pos']:.2f} | Neutral: {vader_scores['neu']:.2f} | Negative: {vader_scores['neg']:.2f}")
                    
                    # Visualization
                    vader_progress_value = (compound_score + 1) / 2  # Convert from [-1,1] to [0,1]
                    st.progress(vader_progress_value)
                
                # Hugging Face results
                with col2:
                    st.subheader("DistilBERT Analysis")
                    hf_result = result["hf_sentiment"]
                    hf_sentiment = hf_result["label"]
                    hf_score = hf_result["score"]
                    
                    score_color = "green" if hf_sentiment == "POSITIVE" else "red"
                    st.markdown(f"**Sentiment:** <span style='color:{score_color};font-weight:bold'>{hf_sentiment}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** <span style='color:{score_color};font-weight:bold'>{hf_score:.4f}</span>", unsafe_allow_html=True)
                    
                    # Visualization
                    st.progress(hf_score)
                  # Display results in a table
                st.subheader("Comparison")
                
                # Get VADER sentiment label for comparison
                vader_compound = result["vader_sentiment"]["compound"]
                if vader_compound >= 0.05:
                    vader_sentiment_label = "POSITIVE"
                elif vader_compound <= -0.05:
                    vader_sentiment_label = "NEGATIVE"
                else:
                    vader_sentiment_label = "NEUTRAL"
                    
                data = {
                    "Model": ["VADER", "DistilBERT"],
                    "Sentiment": [vader_sentiment_label, result["hf_sentiment"]["label"]],
                    "Score": [vader_compound, result["hf_sentiment"]["score"]]
                }
                results_df = pd.DataFrame(data)
                st.table(results_df)

# Instructions
st.markdown("---")
st.subheader("How to use this app")
st.markdown("""
1. Make sure the FastAPI backend is running (execute `uvicorn app.api:app --reload`)
2. Enter your feedback text in the text area above
3. Click the "Analyze Sentiment" button
4. View the sentiment analysis results from both models
""")

# Footer
st.markdown("---")
st.caption("Feedback Sentiment Analyzer © 2025")
