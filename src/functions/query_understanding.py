from src.llm.prompts import Prompts
from sentence_transformers import SentenceTransformer
from src.llm.schemas import AMBIGUOUS_BOOL_SCHEMA, REWRITTEN_QUERY_SCHEMA
from src.llm.schemas import CLARIFYING_QUESTIONS_SCHEMA
import json
from typing import Dict, Any, List

def _get(obj, key, default=None):
    # dict-style
    if isinstance(obj, dict):
        return obj.get(key, default)
    # attribute-style
    return getattr(obj, key, default)

class QueryUnderstanding:
    def __init__(self, llm, config, session_db):
        self.llm = llm 
        self.config = config 
        self.embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.session_db = session_db
    
    def is_ambiguous(self, query: str) -> bool:
        messages = Prompts.ambiguous_bool(query)
        out = self.llm.chat_structured(messages=messages, json_schema=AMBIGUOUS_BOOL_SCHEMA)
        print(out)

        
        return bool(out["is_ambiguous"])
    
    def rewrite_query(self, query: str) -> str:
        prompt = Prompts.rewrite_query_messages(query)
        response = self.llm.chat_structured(messages=prompt, json_schema=REWRITTEN_QUERY_SCHEMA)
        return response["rewritten_query"]
    
    

    def _safe_json_load(self, s: str, default: Any):
        try:
            return json.loads(s) if s else default
        except Exception:
            return default

    def _dedupe_keep_order(self, xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            x = (x or "").strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _truncate_list(self, xs: List[Any], max_n: int) -> List[Any]:
        return xs[:max_n]
    


    def context_augmentation(self, current_messages_window: List[Dict[str, Any]], relevant_fields: List[Any]) -> str:
        """
        Context augmentation = build a structured context object that can be injected into the LLM prompt
        """
        # 1) Keep recent window small + stable shape
        recent_messages = self._truncate_list(current_messages_window, max_n=8)

        # 2) Parse retrieved sessions (Milvus hits)
        retrieved_memories = []
        agg_prefs, agg_constraints, agg_open_qs = [], [], []
        agg_key_facts = []

        for hit in (relevant_fields or [])[:3]:  # top-3 only (avoid prompt bloat)
            # Depending on your Milvus result object, adjust these fields
            pk = getattr(hit, "id", None) or getattr(hit, "pk", None) or ""
            score = getattr(hit, "score", None)

            # You store full_session_json in entity (based on your snippet)
            entity = getattr(hit, "entity", None) or {}
            full_json_str = entity.get("full_session_json", "{}")
            sess = self._safe_json_load(full_json_str, default={})

            ss = sess.get("session_summary", {})
            user_profile = ss.get("user_profile", {}) or {}

            prefs = user_profile.get("prefs", []) or []
            constraints = user_profile.get("constraints", []) or []
            open_qs = ss.get("open_questions", []) or []
            key_facts = ss.get("key_facts", []) or []
            decisions = ss.get("decisions", []) or []
            todos = ss.get("todos", []) or []

            # Aggregate signals
            agg_prefs.extend([str(x) for x in prefs])
            agg_constraints.extend([str(x) for x in constraints])
            agg_open_qs.extend([str(x) for x in open_qs])
            agg_key_facts.extend([str(x) for x in key_facts])

            retrieved_memories.append({
                "pk": pk,
                "score": score,
                "session_id": ss.get("session_id", None),
                "summary_idx": ss.get("summary_idx", None),
                "key_facts": self._truncate_list([str(x) for x in key_facts], 8),
                "decisions": self._truncate_list([str(x) for x in decisions], 5),
                "todos": self._truncate_list([str(x) for x in todos], 5),
            })

        # 3) Dedupe & cap memory signals
        memory_signals = {
            "user_prefs": self._truncate_list(self._dedupe_keep_order(agg_prefs), 10),
            "user_constraints": self._truncate_list(self._dedupe_keep_order(agg_constraints), 10),
            "open_questions": self._truncate_list(self._dedupe_keep_order(agg_open_qs), 10),
        }

        # 4) Build "context_text" for prompt injection (OpenAI cookbook-style: inject retrieved text into messages) :contentReference[oaicite:3]{index=3}
        key_facts_clean = self._truncate_list(self._dedupe_keep_order(agg_key_facts), 12)
        context_text_parts = []
        if memory_signals["user_prefs"]:
            context_text_parts.append("User preferences:\n- " + "\n- ".join(memory_signals["user_prefs"]))
        if memory_signals["user_constraints"]:
            context_text_parts.append("User constraints:\n- " + "\n- ".join(memory_signals["user_constraints"]))
        if memory_signals["open_questions"]:
            context_text_parts.append("Open questions:\n- " + "\n- ".join(memory_signals["open_questions"]))
        if key_facts_clean:
            context_text_parts.append("Relevant memory facts:\n- " + "\n- ".join(key_facts_clean))

        augmented_context: Dict[str, Any] = {
            "recent_messages": recent_messages,
            "memory_signals": memory_signals,
            "retrieved_memories": retrieved_memories,
            "context_text": "\n\n".join(context_text_parts).strip()
        }

        return augmented_context

    def clarifying_questions_generation(self, query: str) -> list:
        messages = Prompts.clarifying_questions_generation(query)
        out = self.llm.chat_structured(messages=messages, json_schema=CLARIFYING_QUESTIONS_SCHEMA)
        return out["questions"]
    
    def load_json(self, json_str: str) -> dict:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to decode JSON string.")
            return {}
    
    

    def analyze_query(self, query: str, current_messages_window: list) -> dict:
        if self.is_ambiguous(query):
            is_ambiguous = True
            rewritten_query = self.rewrite_query(query)
        else:
            is_ambiguous = False
            rewritten_query = query
        
        embedding_rewritten_query = self.get_embedding(rewritten_query)


        # retrieve relevant context from session database
        
        relevant_session = self.session_db.retrieve_relevant_session_memory(embedding_rewritten_query, top_k=self.config['topk'])
        if relevant_session[0] != []:
            top1_session = relevant_session[0]
            full_session_json = _get(top1_session, "full_session_json", "{}")
            session_content = self.load_json(full_session_json) if relevant_session else {}
            user_prefs = session_content.get("session_summary", {}).get("user_profile", {}).get("prefs", [])
            open_questions = session_content.get("session_summary", {}).get("open_questions", [])
        else:
            user_prefs = []
            open_questions = []
        # augment context
        final_augmented_context = self.context_augmentation(current_messages_window, relevant_session[0])
        # generate clarifying questions

        if self.is_ambiguous(rewritten_query):
            clarifying_questions = self.clarifying_questions_generation(rewritten_query)
        else:
            clarifying_questions = []
        

        return {
            "original_query": query,
            "is_ambiguous": is_ambiguous,
            "rewritten_query": rewritten_query,
            "needed_context_from_memory": [user_prefs,
            open_questions],
            "clarifying_questions": clarifying_questions,
            "final_augmented_context": final_augmented_context
        }

    def get_embedding(self, text: str):
        embedding = self.embedding.encode([text])
        return embedding