import streamlit as st
import requests

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---- HEADER ----
st.title("🎬 AI Sentiment Analyzer")
st.caption("Analyze movie reviews using Machine Learning")

st.markdown("---")

# ---- EXAMPLE BUTTONS ----
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

# ---- ANALYZE BUTTON ----
if st.button("🚀 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        try:
            with st.spinner("🔍 Analyzing sentiment..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    params={"text": text}
                )
                result = response.json()

            sentiment = result["prediction"]
            confidence = result["confidence"]

            st.markdown("---")

            # ---- RESULT DISPLAY ----
            if sentiment == "positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            # ---- CONFIDENCE BAR ----
            st.subheader("📊 Confidence Level")
            st.progress(confidence)
            st.write(f"Confidence Score: **{confidence:.2f}**")

        except:
            st.error("❌ API not running. Please start FastAPI first.")

st.markdown("---")

# ---- MODEL INFO ----
with st.expander("🧠 Model Details"):
    st.write("Model: Logistic Regression")
    st.write("Accuracy: 89%")
    st.write("Vectorization: TF-IDF")

# ---- FOOTER ----
st.markdown("---")
st.caption("Built by Akinda | Machine Learning Project")