import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

st.set_page_config(
    page_title="Smart Study Notes Generator",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Smart Study Notes Generator")

st.write(
    "Enter a paragraph and generate a short summary "
    "and key study points using Generative AI."
)


@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


def generate_summary(text):

    prompt = "Summarize the following paragraph in a short and clear way: " + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=20,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


text = st.text_area(
    "Enter your paragraph:",
    height=250,
    placeholder="Paste your study paragraph here..."
)


if st.button("✨ Generate Notes"):

    if not text.strip():

        st.warning("Please enter a paragraph first.")

    else:

        original_words = text.split()
        original_count = len(original_words)

        if original_count < 30:

            st.warning(
                "Please enter a paragraph with at least 30 words."
            )

        else:

            with st.spinner("Generating your study notes..."):

                summary = generate_summary(text)

            summary_words = summary.split()
            summary_count = len(summary_words)

            reduction = (
                (original_count - summary_count)
                / original_count
            ) * 100

            st.subheader("📝 Summary")

            st.write(summary)

            st.subheader("🔑 Key Points")

            sentences = re.split(
                r'(?<=[.!?])\s+',
                summary
            )

            for sentence in sentences[:5]:

                if sentence.strip():

                    st.write("• " + sentence.strip())

            st.subheader("📊 Text Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Original Words",
                    original_count
                )

            with col2:

                st.metric(
                    "Summary Words",
                    summary_count
                )

            with col3:

                st.metric(
                    "Text Reduced",
                    f"{reduction:.2f}%"
                )

            st.success(
                "Study notes generated successfully! 🎉"
            )