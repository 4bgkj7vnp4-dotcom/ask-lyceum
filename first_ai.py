import os 
from dotenv import load_dotenv
from google import genai


load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain in two sentences what a college handbook is for."
)
print(response.text)