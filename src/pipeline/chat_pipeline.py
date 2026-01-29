# src/pipeline/chat_pipeline.py
from src.llm.client import GroqClient
from src.schemas.session_summary import Summarization
import json
import os 

class ChatPipeline:
    def __init__(self, config: dict = None):
        self.config = config
        self.context_length = self.load_chat_history()
        self.llm = GroqClient(config=config)
        

        self.session_summary = Summarization(self.llm, self.config)
    
    def load_chat_history(self):
        # fallback object
        default_state = {
            "chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_context_length": 0,
            "max_context_length": self.config.get("max_context_length", 2048) if self.config else 2048,
            "current_chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_message_window": [],  # list of {"role","content",...}
            "lastest_summary_idx": 0,
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

    


    def chat_formation(self, user_input: dict, return_msg: dict) -> list:
        messages = {}
        messages['user'] = {"role": "user", "content": user_input}
        messages['assistant'] = {"role": "assistant", "content": return_msg['content']}
        messages['idx'] = len(self.context_length['current_message_window'])
        # print(messages)
        return messages
            
    
    def infinite_chat(self):
        print("Start chatting with the LLM (type 'exit' to quit)...")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break
            return_msg = self.llm.chat(user_input)
            all_tokens = return_msg['usage'].total_tokens # get total tokens 
            self.context_length['current_context_length'] += all_tokens

            self.context_length['current_message_window'].append(self.chat_formation(user_input, return_msg))

            if self.context_length['current_context_length'] > self.context_length['max_context_length']:
                print("Context length exceeded maximum limit. Resetting context length.")

                self.session_summary.summarize_session(self.context_length['current_message_window'], self.context_length['lastest_summary_idx'])
                self.context_length['lastest_summary_idx'] += 1

                self.context_length['current_context_length'] = all_tokens  # tokenizer-based based on groq docs
                self.context_length['current_message_window'] = []  # reset the message window
                




            

            print(f"LLM: {return_msg['content']}\n")

            print(f"[Context Length: {self.context_length['current_context_length']}/{self.context_length['max_context_length']}]")

    def chat(self, text: str) -> str:
        return self.llm.chat(text)
