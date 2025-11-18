import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # loads .env

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("API key NOT FOUND. Fix your .env file.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Hello Gemini, respond with YES only.")

print(response.text)
