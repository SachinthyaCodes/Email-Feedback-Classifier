import requests
import json

# Test cases for sentiment analysis
test_cases = [
    "I love this product! It works perfectly and the customer service was excellent.",
    "This is okay, nothing special but it does the job.",
    "Terrible experience. Product arrived broken and customer service was unhelpful."
]

def test_api(text):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": text})
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Text: {text}")
            print(f"VADER: {result['vader_sentiment']} ({result['vader_score']:.4f})")
            print(f"DistilBERT: {result['huggingface_sentiment']} ({result['huggingface_score']:.4f})")
            print("-" * 50)
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Sentiment Analysis API...")
    print("=" * 50)
    
    success = True
    for test in test_cases:
        if not test_api(test):
            success = False
    
    if success:
        print("All tests completed successfully!")
        print("The API is working correctly.")
    else:
        print("Some tests failed. Make sure the API server is running.")
    
    print("=" * 50)
