import json
import os 

from src.llm.prompts import Prompts
from src.llm.schemas import SESSION_SUMMARY_CONTENT_SCHEMA

class Summarization:
    def __init__(self, llm, config):
        self.llm = llm 
        self.chat_id = config['chat_id'] if config and 'chat_id' in config else "default_chat"
        self.config = config
        self.prompt = Prompts.summarization
    
    def _save(self, summary: dict, summary_idx: int):
        os.makedirs(self.config["chat_history_path"], exist_ok=True)
        os.makedirs(os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}"), exist_ok=True)
        save_path = os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}", f"{summary_idx}_summary.json")
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved session summary to {save_path}")


    def summarize_session(self, chat_window: list, summary_idx: int) -> dict:
        start_idx = chat_window[0]["idx"]
        end_idx = chat_window[-1]["idx"]

        messages = Prompts.summarization(chat_window)  # list messages
        content = self.llm.chat_structured(messages, SESSION_SUMMARY_CONTENT_SCHEMA)

        result = {
        "session_summary": {
            "session_id": self.chat_id,
            "summary_idx": summary_idx,
            **content
        },
        "message_range_summarized": {"from": start_idx, "to": end_idx}
        }

        self._save(result, summary_idx)
        return result


    # def summarize_session(self, chat_window: list, summary_idx: int) -> str:
    #     # Placeholder for summarization logic
    #     start_idx = chat_window[0]['idx'] if chat_window else 0
    #     end_idx = chat_window[-1]['idx'] if chat_window else 0
    #     summary_prompt = self.prompt(chat_window)
                
    #     summary = self.llm.chat(summary_prompt)

    #     return_msg = { # illustrated schema
    #         "session_summary": {
    #             "session_id": self.chat_id,
    #             "summary_idx" : summary_idx,
    #             "user_profile": {"prefs": [], "constraints": []},
    #             "key_facts": summary['content'],
    #             "decisions": [],
    #             "open_questions": [],
    #             "todos": []
    #         },
    #         "message_range_summarized": {"from": start_idx, "to": end_idx}
    #     }
    #     os.makedirs(self.config["chat_history_path"], exist_ok=True)
    #     os.makedirs(os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}"), exist_ok=True)
    #     save_path = os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}", f"{summary_idx}_summary.json")
    #     with open(save_path, "w") as f:
    #         json.dump(return_msg, f, indent=4)
    #     print(f"Saved session summary to {save_path}")
    #     return return_msg

