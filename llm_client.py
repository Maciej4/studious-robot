import requests
import json


class MessageHistory:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def add_message_from_json(self, json_message):
        try:
            message = json.loads(json_message)
            if "choices" in message and len(message["choices"]) > 0:
                for choice in message["choices"]:
                    if (
                        "message" in choice
                        and "role" in choice["message"]
                        and "content" in choice["message"]
                    ):
                        self.history.append(
                            {
                                "role": choice["message"]["role"],
                                "content": choice["message"]["content"],
                            }
                        )
            else:
                raise ValueError(
                    "JSON message must contain 'choices' with at least one valid message."
                )
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format.")

    def clear_history(self):
        self.history = []

    def get_history(self):
        return self.history

    def get_last_message_str(self) -> str:
        if self.history is None or len(self.history) == 0:
            raise ValueError("Message history is empty.")

        message_dict = self.history[-1]

        if "content" not in message_dict:
            raise ValueError("Message does not contain 'content' field.")

        return message_dict["content"]

    def get_last_message_role(self) -> str:
        if self.history is None or len(self.history) == 0:
            raise ValueError("Message history is empty.")

        message_dict = self.history[-1]

        if "role" not in message_dict:
            raise ValueError("Message does not contain 'role' field.")

        return message_dict["role"]

    def pretty_print(self):
        for message in self.history:
            print(f"{message['role']}: {message['content']}")

    def __str__(self):
        return json.dumps(self.history, indent=2)


class LLMClient:
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def invoke(self, message_history: MessageHistory) -> MessageHistory:
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "messages": message_history.get_history()}
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            message_history.add_message_from_json(json.dumps(response_data))
        else:
            response.raise_for_status()

        return message_history


if __name__ == "__main__":
    # Example usage
    # Initialize message history
    history = MessageHistory()
    history.add_message("system", "Always answer in rhymes.")
    history.add_message("user", "Introduce yourself.")

    # Add a message from JSON
    # json_message = '{"role": "user", "content": "What is the weather like today?"}'
    # history.add_message_from_json(json_message)

    # Initialize LLM client
    llm_client = LLMClient(
        url="http://localhost:1234/v1/chat/completions",
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    )

    # Invoke LLM with the message history
    updated_history = llm_client.invoke(history)

    # Print the updated message history
    print(updated_history)
