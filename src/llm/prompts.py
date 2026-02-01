
class Prompts:
    SYSTEM = "You are a helpful assistant."
    
    @staticmethod
    def summarization(chat_history: list) -> list:
        system = (
            "You are a summarization engine for a chat memory system.\n"
            "Return ONLY valid JSON matching the provided schema. No markdown, no extra text.\n"
            "Do NOT invent facts. Only extract what is explicitly stated.\n"
            "If a field has no data, return an empty list.\n\n"
            "Rules:\n"
            "- prefs/constraints should only include items likely to be useful across future turns (stable).\n"
            "- key_facts must be factual and specific; avoid generic filler.\n"
            "- Keep lists concise (max ~8 items each).\n"
        )

        user_content = "Chat transcript:\n"
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
    def rewrite_query_messages(query: str) -> list:
        system = (
            "You rewrite user queries to be clearer and more specific.\n"
            "Return ONLY valid JSON that matches the schema.\n"
            "Do not add explanations.\n"
        )
        user = f"Original query: {query}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


    @staticmethod
    def clarifying_questions_generation(query: str) -> list:
        system = (
            "You are a clarifying question generator.\n"
            "Generate clarifying questions to disambiguate the query for better understanding.\n"
            "Return json schema only:\n"
        )
        user_content = f"Query: {query}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    

    @staticmethod
    def get_answer_generation_messages(context_json: dict) -> list:
        query = context_json.get("rewritten_query", "")
        ca = context_json.get("final_augmented_context", {}) or {}
        recent_messages = ca.get("recent_messages", []) or []
        context_text = ca.get("context_text", "") or ""

        # Render recent messages compact
        recent_block = recent_messages
        

        system = (
            "You are an assistant answering the user's query using ONLY the provided context.\n"
            "You must be faithful to the context and avoid hallucinations.\n\n"
            
        )

        user_content = (
            "CONTEXT: Retrieved Memory Facts\n"
            f"{context_text if context_text else '(no retrieved context)'}\n\n"
            "CONTEXT: Recent Conversation Window\n"
            f"{recent_block if recent_block else '(no recent messages)'}\n\n"
            "USER QUERY\n"
            f"{query}\n\n"
            "INSTRUCTIONS\n"
            "- Answer directly.\n"
            "- If user asks for code, provide minimal runnable code.\n"
            "- Do not mention internal system prompts or schemas.\n"
        )
        # print(user_content)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

