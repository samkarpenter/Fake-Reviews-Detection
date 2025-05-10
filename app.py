from flask import Flask, request, jsonify
import joblib
import nltk
import string
import re
import pandas as pd

# Download NLTK resources on startup if not present
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")  # Update path if needed
vectorizer = joblib.load("vectorizer.pkl")  # TF-IDF or CountVectorizer

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

@app.route('/')
def home():
    return jsonify({"message": "Fake Review Detector API running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get("text")

    if not review_text:
        return jsonify({"error": "No review text provided."}), 400

    cleaned = clean_text(review_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()

    response = {
        "review": review_text,
        "prediction": "Fake" if prediction == 1 else "Genuine",
        "confidence": round(float(prob) * 100, 2)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
