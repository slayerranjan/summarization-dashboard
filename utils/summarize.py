import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")

# Configure Gemini
genai.configure(api_key=gemini_key)

# ---------- Gemini summarization ----------
def summarize_gemini(text: str, style: str = "neutral", max_words: int = 150) -> str:
    """
    Summarize using Google Gemini.
    Uses models/gemini-2.5-flash (fast, free tier) by default.
    Prompts are tightened to enforce brevity.
    """

    style_instructions = {
    "neutral": f"Summarize the text in no more than {max_words} words. Focus only on factual details. Avoid emotional or descriptive language.",
    "concise": "Summarize in 3 bullet points, each <= 20 words. Only facts, no elaboration.",
    "layperson": f"Explain for a non-expert in <= {max_words} words, using simple language. Avoid repetition.",
    "policy": f"Summarize highlighting risks, limitations, compliance notes in <= {max_words} words. Keep it factual and short."

    }

    prompt = f"{style_instructions.get(style, style_instructions['neutral'])}\n\n{text}"

    # Use the supported model from your list
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    resp = model.generate_content(prompt)

    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()
    return "Gemini returned no text."

# ---------- Hugging Face summarization ----------
def summarize_hf(text: str, max_words: int = 150) -> str:
    """
    Summarize locally using Hugging Face.
    Requires: pip install transformers torch
    """
    from transformers import pipeline

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    max_tokens = max(64, int(max_words * 1.5))
    res = summarizer(text, max_length=max_tokens, min_length=60, do_sample=False)

    return res[0]["summary_text"].strip()
