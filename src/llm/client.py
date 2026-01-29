# src/llm/client.py
import os
from groq import Groq
from src.llm.prompts import Prompts

class GroqClient:
    def __init__(self, config: dict = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Put it in .env or export GROQ_API_KEY=..."
            )
        self.client = Groq(api_key=api_key)
        self.config = config

    def chat(self, user_text: str, model: str = "allam-2-7b") -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": Prompts.SYSTEM},
                {"role": "user", "content": user_text},
            ],
            temperature=self.config['chatbot_temperature'] if self.config else 0.7,
            max_completion_tokens=self.config['max_completion_tokens'] if self.config else 512,
        )

        return_msg = {
            "role": resp.choices[0].message.role,
            "content": resp.choices[0].message.content,
            "usage": resp.usage,
        }
        return return_msg
