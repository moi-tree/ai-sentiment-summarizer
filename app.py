import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Insights", page_icon="🤖")

st.title("🤖 AI Text Summarizer & Sentiment Analyzer")
st.write("Powered by Hugging Face Transformers & Streamlit")

# Load Hugging Face pipelines with explicit models
@st.cache_resource
def load_models():
    # Explicit sentiment model
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )
    # Explicit summarizer model
    summarizer_model = pipeline(
        "summarization", 
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt"
    )
    return sentiment_model, summarizer_model

sentiment_pipe, summarizer_pipe = load_models()

# User input
user_text = st.text_area("Enter your text or article below:", height=200)

if st.button("Analyze & Summarize"):
    if user_text.strip():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            sentiment = sentiment_pipe(user_text)[0]
            label = sentiment['label']
            score = round(sentiment['score'] * 100, 2)
            st.metric("Tone", label, f"{score}% confidence")
            
        with col2:
            st.subheader("📝 Summary")
            if len(user_text.split()) > 30:
                summary = summarizer_pipe(user_text, max_length=60, min_length=20, do_sample=False)
                st.write(summary[0]['summary_text'])
            else:
                st.info("Provide at least 30 words for a meaningful summary.")
    else:
        st.warning("Please enter some text first!")
