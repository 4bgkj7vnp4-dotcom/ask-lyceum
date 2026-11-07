import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

SYSTEM_PROMPT = (
    "You are Ask Lyceum, a friendly assistant for college students. "
    "Answer clearly and briefly. If you are not sure, say you don't know."
)

history = []


def ask(question):
    history.append({"role": "user", "parts": [{"text": question}]})

    for attempt in range(3):
        for model in MODELS:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=history,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                answer = response.text
                history.append({"role": "model", "parts": [{"text": answer}]})
                return answer
            except Exception as error:
                print(f"[{model}] Problem: {error}")
        time.sleep(5 * (attempt + 1))

    history.pop()
    return "Sorry, the AI service is busy right now. Please try again."


print("Ask Lyceum is ready. Type 'quit' to exit.")

while True:
    question = input("You: ")
    if question.strip().lower() == "quit":
        print("Goodbye!")
        break
    if question.strip() == "":
        continue
    print("Ask Lyceum:", ask(question))