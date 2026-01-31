import os
import nltk
from dotenv import load_dotenv

# --- Ensure NLTK resources are available (Cloud-safe) ---
nltk.download("punkt", quiet=True)

# --- Load local environment variables ---
load_dotenv()

# --- Resolve GEMINI_API_KEY ---
gemini_key = None

try:
    import streamlit as st
    # Cloud: use secrets first (safe get avoids errors if secrets missing)
    gemini_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    # Not running inside Streamlit, ignore
    pass

# Local fallback: .env
if not gemini_key:
    gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Set it in Streamlit Secrets (cloud) or .env (local).")

# --- Configure Gemini (new SDK) ---
import google.genai as genai

# Create a client with the API key (new SDK pattern)
client = genai.Client(api_key=gemini_key)

# ---------- Gemini summarization ----------
def summarize_gemini(text: str, style: str = "neutral", max_words: int = 150) -> str:
    """
    Summarize using Google Gemini (models/gemini-2.5-flash).
    """
    style_instructions = {
        "neutral": f"Summarize the text in no more than {max_words} words. Focus only on factual details. Avoid emotional or descriptive language.",
        "concise": "Summarize in 3 bullet points, each <= 20 words. Only facts, no elaboration.",
        "layperson": f"Explain for a non-expert in <= {max_words} words, using simple language. Avoid repetition.",
        "policy": f"Summarize highlighting risks, limitations, compliance notes in <= {max_words} words. Keep it factual and short."
    }

    prompt = f"{style_instructions.get(style, style_instructions['neutral'])}\n\n{text}"

    model = client.GenerativeModel("models/gemini-2.5-flash")
    resp = model.generate_content(prompt)

    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return "Gemini returned no text."

# ---------- Hugging Face summarization ----------
def summarize_hf(text: str, max_words: int = 150) -> str:
    """
    Summarize using Hugging Face (DistilBART).
    Requires: transformers==4.35.2, torch>=2.0.0, sentencepiece
    """
    from transformers import pipeline

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Convert word target to a safe token length; DistilBART expects tokens
    max_tokens = max(64, int(max_words * 1.5))
    # Cap max_length to avoid warnings if input is shorter
    max_tokens = min(max_tokens, len(text) + 100)

    res = summarizer(text, max_length=max_tokens, min_length=60, do_sample=False)
    return res[0]["summary_text"].strip()
