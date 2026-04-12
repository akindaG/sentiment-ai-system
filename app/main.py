from fastapi import FastAPI
import pickle
import re

app = FastAPI()

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}

@app.post("/predict")
def predict(text: str):
    # Clean input
    cleaned = clean_text(text)

    # Vectorize
    vec = vectorizer.transform([cleaned])

    # Predict
    pred = model.predict(vec)[0]

    # Convert to label
    sentiment = "positive" if pred == 1 else "negative"

    # Confidence score
    proba = model.predict_proba(vec)[0]
    confidence = max(proba)

    return {
        "input": text,
        "prediction": sentiment,
        "confidence": float(confidence)
    }