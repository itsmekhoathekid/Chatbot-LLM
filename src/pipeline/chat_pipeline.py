# src/pipeline/chat_pipeline.py
from src.llm.client import GroqClient
from src.functions.session_summary import Summarization
from src.functions.query_understanding import QueryUnderstanding

import json
import os 
import asyncio

class ChatPipeline:
    def __init__(self, config: dict = None):
        self.config = config
        self.context_length = self.load_chat_history()
        self.llm = GroqClient(config=config)
        

        self.session_summary = Summarization(self.llm, self.config)
    
    def save_chat_history(self):
        if not self.config:
            return
        os.makedirs(self.config["chat_history_path"], exist_ok=True)
        save_path = os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}.json")
        with open(save_path, "w") as f:
            json.dump(self.context_length, f, indent=4)
        print(f"Saved chat history to {save_path}")
    
    def load_chat_history(self):
        # fallback object
        default_state = {
            "chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_context_length": 0,
            "max_context_length": self.config.get("max_context_length", 2048) if self.config else 2048,
            "current_chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_message_window": [],  # list of {"role","content",...}
            "all_messages": [],
            "lastest_summary_idx": 0,
            "latest_summary" : None,
        }

        if not self.config:
            return default_state

        if self.config.get("reload", False):
            save_path = os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}.json")
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    history = json.load(f)
                print(f"Loaded chat history from {save_path}")
                return history

            # reload=True nhưng chưa có file -> fallback
            print(f"[WARN] reload=True but no history found at {save_path}. Starting fresh.")
            return default_state

        return default_state




    def chat_formation(self, user_input: str, return_msg: dict) -> dict:
        return {
            "user": {"role": "user", "content": user_input},
            "assistant": {"role": "assistant", "content": return_msg["content"]},
            "idx": len(self.context_length["all_messages"]),
        }

    async def infinite_chat(self):
        print("Start chatting with the LLM (type 'exit' to quit)...", flush=True)

        while True:
            # input() là blocking -> đưa vào thread để await được
            user_input = await asyncio.to_thread(input, "You: ")
            
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!", flush=True)
                break

            # llm.chat sync -> chạy trong thread, vẫn await được
            return_msg = await asyncio.to_thread(self.llm.chat, user_input)

            all_tokens = return_msg["usage"].total_tokens
            self.context_length["current_context_length"] += all_tokens

            msg_obj = self.chat_formation(user_input, return_msg)
            self.context_length["current_message_window"].append(msg_obj)
            self.context_length["all_messages"].append(msg_obj)

            if self.context_length["current_context_length"] > self.context_length["max_context_length"]:
                print(f"Context length exceeded maximum limit. [{self.context_length['current_context_length']}/{self.context_length['max_context_length']}] Generating summary...", flush=True)

                # summarize_session sync -> chạy trong thread, await cho “đợi xong mới cho nhập”
                summary = await asyncio.to_thread(
                    self.session_summary.summarize_session,
                    self.context_length["current_message_window"],
                    self.context_length["lastest_summary_idx"],
                )

                self.context_length["latest_summary"] = summary
                self.context_length["lastest_summary_idx"] += 1

                self.context_length["current_context_length"] = 0
                self.context_length["current_message_window"] = []
                print("Summary done.\n", flush=True)

            print(f"LLM: {return_msg['content']}\n", flush=True)
            print(
                f"[Context Length: {self.context_length['current_context_length']}/{self.context_length['max_context_length']}]",
                flush=True
            )

        self.save_chat_history()

    def chat(self, text: str):
        return self.llm.chat(text)
