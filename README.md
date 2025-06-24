# 📬 Email Feedback Classifier & Sentiment Analyzer (NLP + Transformers)

This beginner-friendly project classifies email feedback into types like **complaints**, **praise**, or **feature requests**, and analyzes **sentiment** using both **rule-based** and **transformer-based** techniques.

💡 Built as a learning experiment using NLP, scikit-learn, Hugging Face Transformers, and VADER.

---

## 🎥 Demo

🎬 [Watch Demo Video](https://youtu.be/sample-demo-link)

---

## 📁 Project Structure

```
📦email-feedback-classifier
├── data/
│   ├── feedback.csv
│   └── feedback_cleaned.csv
├── distilbert-email-feedback-model/ ← (Not included in repo)
├── notebooks/
├── app/
│   ├── api.py - FastAPI backend for sentiment analysis
│   └── streamlit_app.py - Streamlit frontend for user interaction 
├── results/
├── logs/
├── api_test.py
├── test_api.py
├── run_app.bat
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

> 🚧 I am a **beginner in NLP and machine learning**, and this project was built for learning purposes.  
> The model **might not be highly accurate** or production-ready. Any feedback or improvements are welcome!

## 🛠️ Tech Stack

| Task              | Tool/Library                        |
|-------------------|-------------------------------------|
| Text Processing   | `pandas`, `spaCy`, `regex`          |
| Label Encoding    | `LabelEncoder`                      |
| ML Model          | `LogisticRegression`, `TfidfVectorizer` |
| Transformers      | `DistilBERT`, `transformers`, `Trainer` |
| Sentiment Analysis| `VADER`, `transformers.pipeline`    |
| Deployment        | `FastAPI`, `Streamlit`              |
| Visualization     | `matplotlib`, `seaborn`             |

---

## 📊 Example Dataset

| text                                  | label            |
|--------------------------------------|------------------|
| I love this app!                     | praise           |
| It crashes too much.                 | technical_issue  |
| Can you add export option?           | feature_request  |

---

## 🔧 Installation & Setup

### 1. Clone this repo

```
git clone https://github.com/SachinthyaCodes/Email-Feedback-Classifier.git
cd email-feedback-classifier
```

### 2. Create and activate a virtual environment (optional but recommended)

```
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python -m venv env
source env/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 🚀 How to Use

### Step 1: Preprocess Raw Data

```
python data_preprocessing.py
```
- Cleans text (lowercase, punctuation removal)
- Tokenizes
- Encodes labels
- Saves feedback_cleaned.csv

### Step 2: Train a Baseline Classifier (Logistic Regression)

```
python train_logistic_regression.py
```
- Uses TF-IDF + Logistic Regression
- Displays accuracy and confusion matrix

### Step 3: Fine-tune DistilBERT (Transformer)

```
python train_transformer.py
```
- Loads distilbert-base-uncased
- Fine-tunes it using Hugging Face Trainer
- Saves model to local folder

### Step 4: Add Sentiment Scores

```
python sentiment_analysis.py
```
- Adds vader_score, vader_label
- Adds Hugging Face sentiment label + score
- Saves results to predicted_feedback.csv

### Step 5: Run the Sentiment Analysis Application

Either use the batch file:
```
run_app.bat
```

Or start manually:
```
# Terminal 1 - Start FastAPI backend
uvicorn app.api:app --reload

# Terminal 2 - Start Streamlit frontend
streamlit run app/streamlit_app.py
```

The API will be available at http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Streamlit app: http://localhost:8501

### Step 6: Test the API

```
python test_api.py
```

## ⚠️ Model File Not Included

This repository does not contain the fine-tuned model to avoid large file uploads.

Download the model and place it in the project directory.

## 📈 Sample Output

| text | predicted_label | vader_label | huggingface_sentiment |
|------|----------------|------------|----------------------|
| Love the design! | praise | positive | POSITIVE |
| The app keeps crashing. | technical_issue | negative | NEGATIVE |
| Can we get dark mode? | feature_request | neutral | NEUTRAL |

## 💡 Key Learning Areas

- Text preprocessing with spaCy
- Basic ML vs. Transformers for NLP
- Fine-tuning a Hugging Face model
- Sentiment scoring with rule-based & transformer models
- Building and testing a simple API for predictions

## 🧠 Resources & Credits

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)

## 🙋‍♂️ Author

Sachinthya Lakshitha  
🎓 Final year IT Undergraduate  
[LinkedIn Profile](https://www.linkedin.com/in/sachinthya-lakshitha/)  
📧 sachinthyaofficial@email.com
