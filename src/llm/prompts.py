
class Prompts:
    SYSTEM = "You are a helpful assistant."

    @staticmethod
    def get_summarization_prompt(chat_history: list) -> str:
        prompt = "Summarize the following chat session, give me the context only:\n"
        for message in chat_history:
            
            prompt += f"{message['user']['role']} : {message['user']['content']}\n"
            prompt += f"{message['assistant']['role']} : {message['assistant']['content']}\n"
        # print(f"Generated summarization prompt: {prompt}")
        return prompt

    @staticmethod
    def get_ambiguous_classification_prompt(query: str) -> str:
        return f"Is the following query ambiguous? Answer with 'yes' or 'no'. Query: {query}"
    
    @staticmethod
    def get_rewrite_query_prompt(query: str) -> str:
        return f"Rewrite the following query to be more specific: {query}"