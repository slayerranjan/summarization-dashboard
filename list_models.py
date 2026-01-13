import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Configure Gemini with your API key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")

genai.configure(api_key=gemini_key)

# List all models available to your account
print("Available Gemini models for your API key:\n")
for m in genai.list_models():
    print(f"{m.name} | supports generate_content: {'generateContent' in m.supported_generation_methods}")
