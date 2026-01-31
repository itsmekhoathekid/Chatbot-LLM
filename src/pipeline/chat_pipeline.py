# src/pipeline/chat_pipeline.py
import time
from src.llm.client import GroqClient
from src.functions.session_summary import Summarization
from src.functions.query_understanding import QueryUnderstanding
from src.functions.database import Milvus 

import json
import os 
import asyncio

class ChatPipeline:
    def __init__(self, config: dict = None):
        self.config = config
        self.llm = GroqClient(config=config)
        self.session_database = Milvus(config)
        self.context_length = self._load_state_from_db()
        self.session_summary = Summarization(self.llm, self.config)
        self.query_understanding = QueryUnderstanding(self.llm, self.config, self.session_database)

        # Load from Milvus

        
    
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
    
    def make_pk(self, user_id: str, chat_id: str, session_id: int) -> str:
        return f"{user_id}::{chat_id}::{session_id}"
    
    
    def insert_session_content(self, session_content: str, full_session_json: dict = None):
        user_id = self.config.get('user_id', 'default_user') if self.config else 'default_user'
        chat_id = self.context_length['chat_id']
        session_id = self.context_length['lastest_summary_idx'] - 1  # insert the last summarized session
        pk = self.make_pk(user_id, chat_id, session_id)

        key_facts = '. '.join(session_content)

        vec = self.query_understanding.get_embedding(key_facts)

        # vec đang là np.ndarray (1, dim) hoặc list
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # nếu còn nested [[...]] thì flatten về [...]
        if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
            vec = vec[0]

        data = {
            "pk": pk,
            "user_id": user_id,
            "chat_id": chat_id,
            "session_id": session_id,
            "session_content": key_facts,
            "embedding": vec,
            "full_session_json": json.dumps(full_session_json)
        }

        self.session_database.insert(collection_name=self.config["session_collection_name"], data=data)
    
        

    def insert_chat_log(self, user_id: str, chat_id: str, idx: int, role: str, content: str):
        name = self.config["chat_logs_collection_name"]
        pk = f"{user_id}::{chat_id}::{idx}::{role}"
        row = {
            "pk": pk,
            "user_id": user_id,
            "chat_id": chat_id,
            "idx": int(idx),
            "role": role,
            "content": content,
            "created_at": int(time.time()),
        }
        return self.session_database.insert(collection_name=name, data=row)
    
    
    

    def load_chat_logs(self, user_id: str, chat_id: str, limit: int = 200):
        name = self.config["chat_logs_collection_name"]
        expr = f'user_id == "{user_id}" and chat_id == "{chat_id}"'
        # self.session_database.ensure_loaded(name)
        rows = self.session_database.client.query(
            collection_name=name,
            filter=expr,
            output_fields=["idx", "role", "content", "created_at"],
            limit=limit,
        )
        # sort by idx just in case
        rows.sort(key=lambda r: r["idx"])
        return rows
    

    def save_context_window_state(
        self,
        user_id: str,
        chat_id: str,
        current_context_length: int,
        lastest_summary_idx: int,
        window_obj: dict,
        latest_summary_obj: dict | None = None,
    ):
        name = self.config["context_window_collection_name"]
        pk = f"{user_id}::{chat_id}"

        

        row = {
            "pk": pk,
            "user_id": user_id,
            "chat_id": chat_id,
            "current_context_length": int(current_context_length),
            "lastest_summary_idx": int(lastest_summary_idx),
            "window_json": json.dumps(window_obj, ensure_ascii=False),
            "latest_summary_json": json.dumps(latest_summary_obj, ensure_ascii=False) if latest_summary_obj else "{}",
        }
        return self.session_database.insert(collection_name=name, data=row)

    def load_context_window_state(self, user_id: str, chat_id: str) -> dict | None:
        name = self.config["context_window_collection_name"]
        self.session_database.ensure_loaded(name)
        pk = f"{user_id}::{chat_id}"
        rows = self.session_database.client.query(
            collection_name=name,
            filter=f'pk == "{pk}"',
            output_fields=[
                "current_context_length", "lastest_summary_idx", "window_json", "latest_summary_json"
        ])

        # "current_context_length", "lastest_summary_idx", "window_json", "latest_summary_json"
        
        if not rows:
            return None
        r = rows[0]
        return {
            "current_context_length": int(r.get("current_context_length", 0)),
            "lastest_summary_idx": int(r.get("lastest_summary_idx", 0)),
            "current_message_window": json.loads(r.get("window_json", "[]") or "[]"),
            "latest_summary": json.loads(r.get("latest_summary_json", "{}") or "{}"),
        }

    def _default_state(self):
        return {
            "chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_context_length": 0,
            "max_context_length": self.config.get("max_context_length", 2048) if self.config else 2048,
            "current_chat_id": self.config.get("chat_id", "default_chat") if self.config else "default_chat",
            "current_message_window": [],
            "all_messages": [],
            "lastest_summary_idx": 0,
            "latest_summary": None,
            "logs": [],
        }

    def _load_state_from_db(self):
        default_state = self._default_state()

        if not self.config:
            return default_state

        user_id = self.config.get("user_id", "default_user")
        chat_id = self.config.get("chat_id", "default_chat")

        state = self.load_context_window_state(user_id, chat_id)

        # ✅ nếu chưa có state trong DB -> return default (KHÔNG return None)
        if not state:
            return default_state

        # merge
        default_state["current_context_length"] = state.get("current_context_length", 0)
        default_state["lastest_summary_idx"] = state.get("lastest_summary_idx", 0)
        default_state["current_message_window"] = state.get("current_message_window", [])
        default_state["latest_summary"] = state.get("latest_summary", None)

        # load logs (optional)
        try:
            default_state["logs"] = self.load_chat_logs(user_id, chat_id, limit=500)
        except Exception:
            default_state["logs"] = []

        return default_state





    async def infinite_chat(self):
        print("Start chatting with the LLM (type 'exit' to quit)...", flush=True)

        while True:
            # input() là blocking -> đưa vào thread để await được
            user_input = await asyncio.to_thread(input, "You: ")
            
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!", flush=True)
                break
            
            query_understanding_result = await asyncio.to_thread(self.query_understanding.analyze_query, user_input, self.context_length["current_message_window"])
            
            # llm.chat sync -> chạy trong thread, vẫn await được
            return_msg = await asyncio.to_thread(self.llm.query_understanding, query_understanding_result)

            all_tokens = return_msg["usage"].completion_tokens + len(query_understanding_result["rewritten_query"].split()) 

            self.context_length["current_context_length"] += all_tokens

            msg_obj = self.chat_formation(user_input, return_msg)

            # fall back insert chat logs
            self.context_length["current_message_window"].append(msg_obj)
            self.context_length["all_messages"].append(msg_obj)

            # insert chat log into Milvus
            user_id = self.config.get("user_id", "default_user")
            chat_id = self.context_length["chat_id"]
            idx = msg_obj["idx"]
            await asyncio.to_thread(
                self.insert_chat_log,
                user_id,
                chat_id,
                idx,
                "user",
                user_input
            )
            await asyncio.to_thread(
                self.insert_chat_log,
                user_id,
                chat_id,
                idx,
                "assistant",
                return_msg["content"]
            )

            # save context window state into Milvus
            await asyncio.to_thread(
                self.save_context_window_state,
                user_id,
                chat_id,
                self.context_length["current_context_length"],
                self.context_length["lastest_summary_idx"],
                self.context_length["current_message_window"],
                self.context_length["latest_summary"],
            )


            if self.context_length["current_context_length"] > self.context_length["max_context_length"]:
                print(f"Context length exceeded maximum limit. [{self.context_length['current_context_length']}/{self.context_length['max_context_length']}] Generating summary...", flush=True)

                # summarize_session sync -> chạy trong thread, await cho “đợi xong mới cho nhập”
                summary = await asyncio.to_thread(
                    self.session_summary.summarize_session,
                    self.context_length["current_message_window"],
                    self.context_length["lastest_summary_idx"],
                )

                await asyncio.to_thread(
                    self.insert_session_content,
                    summary["session_summary"]["key_facts"],
                    summary
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
