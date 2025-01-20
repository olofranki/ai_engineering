# pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
import sys
import os
from dotenv import load_dotenv

# Dodaj ścieżkę bezpośrednio do katalogu src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
print("sys.path:", sys.path)

from src.ai_eng.service.openai_service import OpenAIService

load_dotenv()

# -----------------------------------------------------------------------------
# Define clients
# -----------------------------------------------------------------------------

openai_service = OpenAIService()
ollama_service = OpenAIService(provider="ollama")
groq_service = OpenAIService(provider="groq")

# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

def get_response_from_groq(user_message: str) -> str:
    response = groq_service.create_chat_completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_service.create_chat_completion(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    message = "What is the capital of Japan?"
    print(get_response_from_ollama(message))

if __name__ == "__main__":
    main()
