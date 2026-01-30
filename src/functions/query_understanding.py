from src.llm.prompts import Prompts
from sentence_transformers import SentenceTransformer
from src.llm.schemas import AMBIGUOUS_BOOL_SCHEMA

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
        prompt = Prompts.get_rewrite_query_prompt(query)
        response = self.llm.chat(prompt)
        return response["content"]
    
    

    def context_augmentation(self, current_messages_window: list, relevant_fields) -> str:
        # Placeholder for context augmentation logic
        # Combine current messages with relevant fields from the database
        augmented_context = {
            "current_messages_window": current_messages_window,
            "relevant_fields": relevant_fields
        }
        return str(augmented_context)

    def clarifying_questions_generation(self, query: str) -> list:
        pass 

    def analyze_query(self, query: str, current_messages_window: list) -> dict:
        if self.is_ambiguous(query):
            is_ambiguous = True
            rewritten_query = self.rewrite_query(query)
        else:
            is_ambiguous = False
            rewritten_query = query
        
        embedding_rewritten_query = self.get_embedding(rewritten_query)


        # retrieve relevant context from session database
        relevant_fields = self.session_db.retrieve_relevant(embedding_rewritten_query, top_k=self.config['topk'])
        # augment context
        final_augmented_context = self.context_augmentation(current_messages_window, relevant_fields)
        # generate clarifying questions

        if self.is_ambiguous(rewritten_query):
            clarifying_questions = self.clarifying_questions_generation(rewritten_query)
        else:
            clarifying_questions = []


        return {
            "original_query": query,
            "is_ambiguous": is_ambiguous,
            "rewritten_query": rewritten_query,
            "needed_context_from_memory": ["user_profile.prefs",
            "open_questions"],
            "clarifying_questions": clarifying_questions,
            "final_augmented_context": final_augmented_context
        }

    def get_embedding(self, text: str):
        embedding = self.embedding.encode([text])
        return embedding