import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="AI Text Insights", page_icon="🤖")

st.title("🤖 AI Text Summarizer & Sentiment Analyzer")
st.write("Powered by Hugging Face Transformers & Streamlit")


# Load Hugging Face models reliably using explicit classes
@st.cache_resource
def load_models():
    # 1. Sentiment Analysis Pipeline
    sentiment_pipe = pipeline(
        task="text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

    # 2. Summarizer using explicit tokenizer & model (prevents KeyError on pipeline)
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return sentiment_pipe, summarizer_tokenizer, summarizer_model


sentiment_pipe, sum_tokenizer, sum_model = load_models()

# User input
user_text = st.text_area("Enter your text or article below:", height=200)

if st.button("Analyze & Summarize"):
    if user_text.strip():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Analysis")
            sentiment = sentiment_pipe(user_text)[0]
            label = sentiment["label"]
            score = round(sentiment["score"] * 100, 2)
            st.metric("Tone", label, f"{score}% confidence")

        with col2:
            st.subheader("📝 Summary")
            word_count = len(user_text.split())

            if word_count > 30:
                # Tokenize input text and generate summary directly
                inputs = sum_tokenizer(
                    "summarize: " + user_text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                )
                summary_ids = sum_model.generate(
                    inputs["input_ids"],
                    max_length=80,
                    min_length=20,
                    do_sample=False,
                )
                summary_text = sum_tokenizer.decode(
                    summary_ids[0], skip_special_tokens=True
                )
                st.write(summary_text)
            else:
                st.info("Provide at least 30 words for a meaningful summary.")
    else:
        st.warning("Please enter some text first!")
