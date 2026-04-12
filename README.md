# 🚀 AI-Powered Sentiment Analysis System

## 📌 Overview

This project implements an end-to-end Machine Learning pipeline that analyzes text sentiment and serves predictions through a production-ready API. The system is containerized using Docker, making it deployable in real-world environments.

---

## 🎯 Problem

Organizations receive large volumes of customer feedback (reviews, comments, social media posts), but manually analyzing sentiment is inefficient and not scalable.

---

## 💡 Solution

This system automates sentiment classification by:

* Cleaning and preprocessing raw text data
* Converting text into numerical features using TF-IDF
* Training machine learning models for classification
* Deploying a REST API using FastAPI
* Containerizing the application using Docker

---

## 🧠 Models Used

* Logistic Regression
* Naive Bayes

---

## 📊 Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | ~88%     | ~0.87    |
| Naive Bayes         | ~85%     | ~0.84    |

> Evaluation performed using Precision, Recall, and F1-score on test data.

---

## 🏗️ System Architecture

```
User Input → API (FastAPI) → Preprocessing → Model → Prediction → Response
```

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Pandas
* FastAPI
* Streamlit
* Docker

---

## 📁 Project Structure

```
sentiment-ai-project/
│
├── data/                # Dataset
├── notebooks/           # Data exploration & experiments
├── src/                 # Training scripts
├── app/                 # API and frontend
├── model.pkl            # Trained model
├── vectorizer.pkl       # TF-IDF vectorizer
├── Dockerfile           # Container configuration
├── requirements.txt     # Dependencies
└── README.md
```

---

## ▶️ Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run API

```bash
uvicorn app.main:app --reload
```

### 3. Access API

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🎨 Run Frontend (Streamlit)

```bash
streamlit run app/ui.py
```

---

## 🐳 Run with Docker

### Build image

```bash
docker build -t sentiment-app .
```

### Run container

```bash
docker run -p 8000:8000 sentiment-app
```

---

## 📸 Screenshots (Add Yours)

* FastAPI Swagger UI (`/docs`)
  <img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/c6bd9a34-14f0-4eec-8752-4e8a9d72eaf1" />
  <img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/72b361f6-2dab-42df-934d-d702a2512a0c" />
  <img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/85a5239c-e53e-44c5-a4d0-683c52974f3b" />



* Streamlit Interface
  <img width="1919" height="1027" alt="image" src="https://github.com/user-attachments/assets/0c6c6113-3d67-4ea0-a6b3-88f006cb44fc" />
  <img width="1919" height="1024" alt="image" src="https://github.com/user-attachments/assets/b41d1ecb-c95a-43bc-9dc5-298301b9ebdb" />
  <img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/f48654a5-5bb6-48d6-b7a2-d1e0abd8dcfc" />
  



---

## 📌 Future Improvements

* Implement Deep Learning models (LSTM / BERT)
* Deploy to cloud platforms (AWS / GCP / Azure)
* Add real-time streaming sentiment analysis
* Improve model performance with advanced NLP techniques

---

## 🧠 Key Learning Outcomes

* Built an end-to-end ML pipeline
* Understood text vectorization (TF-IDF)
* Compared multiple ML models
* Deployed model using FastAPI
* Containerized application using Docker

---

## 📎 Author

**Akinda**
Data Science Student | Aspiring ML Engineer
