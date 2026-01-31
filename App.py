import streamlit as st
import importlib.util
import pathlib
import pandas as pd
import nltk

# ---------- Ensure NLTK resources ----------
# This block ensures both punkt and punkt_tab are available in Streamlit Cloud
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# ---------- Load summarize.py ----------
summarize_path = pathlib.Path(__file__).parent / "utils" / "summarize.py"
spec = importlib.util.spec_from_file_location("summarize", summarize_path)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)

# ---------- Load metrics.py ----------
metrics_path = pathlib.Path(__file__).parent / "utils" / "metrics.py"
spec_m = importlib.util.spec_from_file_location("metrics", metrics_path)
metrics = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(metrics)

# ---------- Page config ----------
st.set_page_config(page_title="LLM Summarization Dashboard", layout="wide")

# ---------- Global styles ----------
st.markdown(
    """
    <style>
    .io-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 0.75rem;
        white-space: pre-wrap;
        font-size: 1rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='margin-bottom: 0;'>LLM Summarization Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Summarize text and evaluate quality with ROUGE, BLEU, and readability.")

# ---------- Sidebar ----------
st.sidebar.header("Evaluation")
eval_enabled = st.sidebar.checkbox("Provide reference summary for metric scoring")
ref_summary = st.sidebar.text_area("Paste reference summary (optional for ROUGE/BLEU)", height=150) if eval_enabled else ""

st.sidebar.markdown("---")
st.sidebar.caption("Built by Ranjan K. Shettigar")

# ---------- Input ----------
st.markdown("### Input")
text_input = st.text_area("Paste your text here", height=300)

uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# ---------- Engine and style ----------
engine = st.selectbox("Summarization engine", ["Hugging Face (BART)", "Gemini"])
style = st.selectbox("Summary style", ["neutral", "concise", "layperson", "policy"])
max_words = st.slider("Max words", min_value=50, max_value=500, value=150, step=10)

# ---------- Generate summary ----------
if st.button("Generate summary"):
    try:
        if not text_input.strip():
            st.warning("Please paste input text before generating a summary.")
        else:
            if engine.startswith("Hugging Face"):
                summary = summarize.summarize_hf(text_input, max_words=max_words)
            else:
                summary = summarize.summarize_gemini(text_input, style=style, max_words=max_words)

            # Generated summary
            st.markdown("#### Generated Summary")
            st.markdown(f"<div class='io-box'>{summary}</div>", unsafe_allow_html=True)

            # ---------- End-user metrics ----------
            st.markdown("#### End-User Metrics (Input vs Summary)")
            comp = metrics.compression_ratio(text_input, summary)
            readability = metrics.readability_score(summary)
            entities = metrics.entity_retention(text_input, summary)

            st.write(f"**Compression Ratio (%):** {comp}")
            st.write(f"**Readability (Flesch-Kincaid Grade):** {readability}")
            st.write(f"**Entity Retention (%):** {entities}")

            # ---------- Reference evaluation scores ----------
            if eval_enabled:
                st.markdown("#### Reference Evaluation Scores")
                if not ref_summary.strip():
                    st.info("Reference summary is empty. Paste a reference to compute ROUGE and BLEU.")
                else:
                    from rouge_score import rouge_scorer
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    rouge_scores = scorer.score(ref_summary, summary)

                    smoothie = SmoothingFunction().method1
                    ref_tokens = ref_summary.split()
                    sum_tokens = summary.split()
                    bleu_score = sentence_bleu([ref_tokens], sum_tokens, smoothing_function=smoothie)

                    scores_card = f"""
                    <div class="io-box">
                        <div><b>ROUGE-1 F1:</b> {rouge_scores['rouge1'].fmeasure:.3f}</div>
                        <div><b>ROUGE-L F1:</b> {rouge_scores['rougeL'].fmeasure:.3f}</div>
                        <div><b>BLEU:</b> {bleu_score:.3f}</div>
                    </div>
                    """
                    st.markdown(scores_card, unsafe_allow_html=True)

            # Original text
            st.markdown("---")
            st.markdown("#### Original Text")
            st.markdown(f"<div class='io-box'>{text_input}</div>", unsafe_allow_html=True)

            # ---------- Download button ----------
            st.markdown("#### Export Results")
            data = {
                "Original Text": [text_input],
                "Summary": [summary],
                "Compression": [comp],
                "Readability": [readability],
                "Entity Retention": [entities]
            }
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)

            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="summary_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error generating summary: {e}")
