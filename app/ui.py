import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🎬")

st.title("🎬 AI Sentiment Analyzer")
st.caption("Analyze movie reviews using Machine Learning")

st.markdown("---")

# Example buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Positive Example"):
        st.session_state["text"] = "This movie was absolutely amazing with great acting"

with col2:
    if st.button("😞 Negative Example"):
        st.session_state["text"] = "This movie was terrible and a waste of time"

# Input
text = st.text_area(
    "✍️ Enter your review:",
    value=st.session_state.get("text", ""),
    height=150
)

# Predict
if st.button("🚀 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = max(proba)

            sentiment = "positive" if pred == 1 else "negative"

            st.markdown("---")

            if sentiment == "positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            st.subheader("📊 Confidence")
            st.progress(float(confidence))
            st.write(f"Confidence Score: {confidence:.2f}")

st.markdown("---")

with st.expander("🧠 Model Details"):
    st.write("Model: Logistic Regression")
    st.write("Accuracy: 89%")
    st.write("Vectorization: TF-IDF")

st.markdown("---")
st.caption("Built by Akinda")