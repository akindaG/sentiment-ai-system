from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    sentiment = "positive" if pred == 1 else "negative"

    return {
        "input": text,
        "prediction": sentiment
    }