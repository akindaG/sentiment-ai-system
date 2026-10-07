<div align="center">

# Sentiment AI System

### End-to-end customer feedback classification with FastAPI, Streamlit, and Docker

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## Overview

This repository demonstrates a compact **machine-learning inference system**, not just a notebook. It covers text cleaning, TF-IDF feature extraction, model training, serialized inference artifacts, a REST API, a Streamlit interface, and containerization.

The current training script uses the **IMDB review dataset** and trains a Logistic Regression classifier.

## System flow

```text
Raw review text
      │
      ▼
Text cleaning
      │
      ▼
TF-IDF vectorizer
      │
      ▼
Logistic Regression
      │
      ▼
Serialized artifacts
      │
      ├── FastAPI inference endpoint
      │
      └── Streamlit user interface
      │
      ▼
   Prediction
```

## Repository structure

```text
sentiment-ai-system/
├── app/
│   ├── main.py          # FastAPI inference service
│   ├── ui.py            # Streamlit interface
│   ├── model.pkl        # Serialized classifier
│   └── vectorizer.pkl   # Serialized TF-IDF vectorizer
├── data/                # Dataset files
├── notebooks/           # Exploration / experiments
├── src/
│   └── train.py         # Training pipeline
├── Dockerfile
├── requirements.txt
└── runtime.txt
```

## What this project demonstrates

| Area | Evidence |
| --- | --- |
| **Preprocessing** | Lowercasing, HTML removal, non-letter cleanup |
| **Feature engineering** | TF-IDF with a bounded feature space |
| **Modeling** | Logistic Regression binary sentiment classification |
| **Artifact persistence** | Pickled classifier + vectorizer |
| **Serving** | FastAPI application |
| **User interface** | Streamlit application |
| **Packaging** | Dockerfile + Python requirements |

## Run locally

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### 3. Run the Streamlit UI

```bash
streamlit run app/ui.py
```

## Train the model

The current training entry point is:

```bash
python src/train.py
```

It loads `data/IMDB_dataset.csv`, cleans reviews, creates a train/test split, fits a TF-IDF vectorizer, trains Logistic Regression, and serializes the resulting artifacts.

> The repository previously documented approximate comparison metrics for Logistic Regression and Naive Bayes. The current checked-in training script does not reproduce that full comparison, so this README intentionally avoids presenting those historical numbers as verified results.

## Docker

```bash
docker build -t sentiment-ai-system .
docker run -p 8000:8000 sentiment-ai-system
```

## Visual evidence

### FastAPI

<img width="100%" alt="FastAPI Swagger interface" src="https://github.com/user-attachments/assets/c6bd9a34-14f0-4eec-8752-4e8a9d72eaf1" />

### Streamlit

<img width="100%" alt="Streamlit sentiment interface" src="https://github.com/user-attachments/assets/4ad6c284-1cbc-4861-bf4a-81c3c1a7c6e3" />

## Engineering improvements I would make next

- add deterministic train/test splitting with `random_state`
- record accuracy, precision, recall, F1, and confusion matrix during training
- version model artifacts instead of committing them as opaque binaries
- add automated API tests
- add CI for linting/tests
- add model metadata and reproducibility information
- deploy the API and UI with a verified public endpoint

---

**Portfolio role:** compact ML deployment project demonstrating the path from text data to a served prediction system.
