import streamlit as st
from transformers import pipeline

# 1. Setup Webpage
st.set_page_config(page_title="AI Text Insights", page_icon="🤖")
st.title("🤖 AI Text Summarizer & Sentiment Analyzer")
st.write("Powered by Hugging Face Transformers & Streamlit")

# 2. Load AI Models (Cached so they don't reload on every button click)
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment_model, summarizer_model
sentiment_pipe, summarizer_pipe = load_models()

# 3. Build User Interface
user_text = st.text_area("Enter your text or article below (minimum 30 words):", height=200)

if st.button("Analyze & Summarize"):
    if user_text.strip():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            # Run sentiment pipeline
            sentiment = sentiment_pipe(user_text)[0]
            label = sentiment['label']
            score = round(sentiment['score'] * 100, 2)
            st.metric("Tone", label, f"{score}% confidence")
            
        with col2:
            st.subheader("📝 Summary")
            # Run summarizer pipeline
            if len(user_text.split()) > 30:
                summary = summarizer_pipe(user_text, max_length=60, min_length=20, do_sample=False)
                st.write(summary[0]['summary_text'])
            else:
                st.info("Please provide at least 30 words for a meaningful summary.")
    else:
        st.warning("Please enter some text first!")
