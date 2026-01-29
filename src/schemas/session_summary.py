import json
import os 

from src.llm.prompts import Prompts

class Summarization:
    def __init__(self, llm, config):
        self.llm = llm 
        self.chat_id = config['chat_id'] if config and 'chat_id' in config else "default_chat"
        self.config = config
        self.prompt = Prompts.get_summarization_prompt

    def summarize_session(self, chat_history: list, summary_idx) -> str:
        # Placeholder for summarization logic
        summary_prompt = self.prompt(chat_history)
                
        summary = self.llm.chat(summary_prompt)

        return_msg = { # illustrated schema
            "session_summary": {
                "session_id": self.chat_id,
                "summary_idx" : summary_idx,
                "user_profile": {"prefs": [], "constraints": []},
                "key_facts": summary['content'],
                "decisions": [],
                "open_questions": [],
                "todos": []
            },
            "message_range_summarized": {"from": 0, "to": len(chat_history) - 1}
        }
        os.makedirs(self.config["chat_history_path"], exist_ok=True)
        os.makedirs(os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}"), exist_ok=True)
        save_path = os.path.join(self.config["chat_history_path"], f"{self.config['chat_id']}", f"{summary_idx}_summary.json")
        with open(save_path, "w") as f:
            json.dump(return_msg, f, indent=4)
        print(f"Saved session summary to {save_path}")
        return return_msg

