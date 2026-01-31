# src/llm/client.py
import os
from groq import Groq
from src.llm.prompts import Prompts

import json 
from typing import Dict, Any, List

class GroqClient:
    def __init__(self, config: dict = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Put it in .env or export GROQ_API_KEY=..."
            )
        self.client = Groq(api_key=api_key)
        self.config = config
        self.max_retries = self.config.get("groq_max_retries", 3) if self.config else 2

    def chat(self, user_text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.config['model_name'] if self.config else "meta-llama/llama-4-scout-17b-16e-instruct",
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
    
    def query_understanding(self, messages: List[Dict[str, str]]) -> Dict[str, Any]: 

        messages = Prompts.get_answer_generation_messages(messages)
        resp = self.client.chat.completions.create(
            model=self.config['model_name'] if self.config else "meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=self.config['chatbot_temperature'] if self.config else 0.7,
            max_completion_tokens=self.config['max_completion_tokens'] if self.config else 512,
        )

        return_msg = {
            "role": resp.choices[0].message.role,
            "content": resp.choices[0].message.content,
            "usage": resp.usage,
        }
        return return_msg
    
    def _ensure_json_contract(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        contract = {
            "role": "system",
            "content": "Return ONLY valid JSON that matches the provided schema. No extra keys, no text."
        }
        return [contract, *messages]

    def chat_structured(self, messages, json_schema) -> Dict[str, Any]:
        messages = self._ensure_json_contract(messages)
        model = self.config["model_name"] if self.config else "meta-llama/llama-4-scout-17b-16e-instruct"

        last_err = None

        # --- 1) strict json_schema path ---
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": json_schema},
                    temperature=self.config['chatbot_temperature'] if self.config else 0.0,
                    max_completion_tokens=self.config.get("max_completion_tokens", 256) if self.config else 256,
                )

                parsed = completion.choices[0].message.content

                # IMPORTANT: parsed thường là dict rồi. Không json.loads.
                if isinstance(parsed, dict):
                    return parsed

                # đôi khi SDK trả string JSON -> parse
                if isinstance(parsed, str):
                    return json.loads(parsed)

                raise TypeError(f"Unexpected parsed type: {type(parsed)}")

            except Exception as e:
                last_err = e

        # --- 2) fallback json_object + normalize + validate ---
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages + [{
                    "role": "user",
                    "content": "Output JSON object only."
                }],
                response_format={"type": "json_object"},
                temperature=self.config['chatbot_temperature'] if self.config else 0.0,
                max_completion_tokens=self.config.get("max_completion_tokens", 256) if self.config else 256,
            )
            raw = completion.choices[0].message.content
            obj = json.loads(raw)

            return obj

        except Exception as e:
            raise RuntimeError(f"Structured output failed. Last strict err={last_err}, fallback err={e}") from e

        
    
    


    


    
