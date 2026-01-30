from src.llm.prompts import Prompts

class QueryUnderstanding:
    def __init__(self, llm, config ):
        self.llm = llm 
        self.config = config 
    
    def ambiguous_classification(self, query: str) -> bool:
        prompt = Prompts.get_ambiguous_classification_prompt(query)
        response = self.llm.chat(prompt)
        if "yes" in response['content'].lower():
            return True
        return False

    
    def rewrite_query(self, query: str) -> str:
        prompt = Prompts.get_rewrite_query_prompt(query)
        response = self.llm.chat(prompt)
        return response['content']
    
    
    

    def context_augmentation(self, query: str) -> str:
        pass  

    def clarifying_questions_generation(self, query: str) -> list:
        pass 

    def analyze_query(self, query: str) -> dict:
        if self.ambiguous_classification(query):
            is_ambiguous = True
            rewritten_query = self.rewrite_query(query)
        else:
            is_ambiguous = False
            rewritten_query = query
        


        return {
            "original_query": query,
            "is_ambiguous": is_ambiguous,
            "rewritten_query": rewritten_query,
            "needed_context_from_memory": ["user_profile.prefs",
            "open_questions"],
            "clarifying_questions": ["...", "..."],
            "final_augmented_context": "..."
        }
