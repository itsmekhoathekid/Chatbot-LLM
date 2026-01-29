
class Prompts:
    SYSTEM = "You are a helpful assistant."

    @staticmethod
    def get_summarization_prompt(chat_history: list) -> str:
        prompt = "Summarize the following chat session:\n"
        for message in chat_history:
            
            prompt += f"{message['user']['role']} : {message['user']['content']}\n"
            prompt += f"{message['assistant']['role']} : {message['assistant']['content']}\n"
        print(f"Generated summarization prompt: {prompt}")
        return prompt