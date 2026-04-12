import streamlit as st
import pickle
import re
import os

# ---- LOAD MODEL SAFELY ----
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "..", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ---- TEXT CLEANING ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---- HEADER ----
st.title("🎬 AI Sentiment Analyzer")
st.caption("Analyze movie reviews using Machine Learning")

st.markdown("---")

# ---- EXAMPLES ----
st.subheader("💡 Try an example")

col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Positive Example"):
        st.session_state["text"] = "This movie was absolutely amazing with great acting"

with col2:
    if st.button("😞 Negative Example"):
        st.session_state["text"] = "This movie was terrible and a waste of time"

st.markdown("---")

# ---- INPUT ----
text = st.text_area(
    "✍️ Enter your review:",
    value=st.session_state.get("text", ""),
    height=150,
    placeholder="Type a movie review here..."
)

# ---- ANALYZE ----
if st.button("🚀 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("🔍 Analyzing sentiment..."):
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = max(proba)

            sentiment = "positive" if pred == 1 else "negative"

            st.markdown("---")

            # RESULT
            if sentiment == "positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            # CONFIDENCE
            st.subheader("📊 Confidence Level")
            st.progress(float(confidence))
            st.write(f"Confidence Score: **{confidence:.2f}**")

st.markdown("---")

# ---- MODEL INFO ----
with st.expander("🧠 Model Details"):
    st.write("Model: Logistic Regression")
    st.write("Accuracy: 89%")
    st.write("Vectorization: TF-IDF")

st.markdown("---")

# ---- FOOTER ----
st.caption("Built by Akinda | Machine Learning Project")