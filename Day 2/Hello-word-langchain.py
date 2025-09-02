# without langchain
import google.generativeai as genai

genai.configure(api_key="your-key")
model = genai.GenerativeModel('gemini-pro')

# You have to manually handle everything
email = "Hey, thanks for the proposal. Can we discuss pricing?"
prompt = f"Generate a professional response to: {email}"
response = model.generate_content(prompt)
print(response.text)  # Might fail, no error handling

# with langchain

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Same interface works with ANY LLM (OpenAI, Claude, etc.)
response = llm.invoke([("human", "Generate a professional response to: Hey, thanks for the proposal. Can we discuss pricing?")])
print(response.content)
