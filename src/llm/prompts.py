
class Prompts:
    SYSTEM = "You are a helpful assistant."
    
    @staticmethod
    def summarization(chat_history: list) -> list:
        """
        Return messages[] instead of raw string
        """
        system = (
            "You are a summarization engine.\n"
            "Extract only factual information from the chat.\n"
            "Do not invent facts.\n"
            "If a field has no data, return an empty list.\n"
            "Return json schema only:\n"
        )

        user_content = "Chat:\n"
        for msg in chat_history:
            user_content += f"user: {msg['user']['content']}\n"
            user_content += f"assistant: {msg['assistant']['content']}\n"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def ambiguous_bool(query: str) -> list:
        return [
            {
                "role": "system",
                "content": (
                    "You are a query ambiguity classifier. "
                    "Return whether the json schema of the query is ambiguous for execution."
                ),
            },
            {"role": "user", "content": f"Query: {query}"},
        ]
    
    @staticmethod
    def get_rewrite_query_prompt(query: str) -> str:
        return f"Rewrite the following query to be more specific: {query}"