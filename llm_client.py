import requests
import json


class MessageHistory:
    def __init__(self):
        self.history = []

    def add(self, role: str, content: str):
        """
        This function adds a message to the message history
        :param role: The role of the message (e.g. system, user, assistant)
        :param content: The content of the message
        """
        self.history.append({"role": role, "content": content})

    def add_message_from_json(self, json_message):
        """
        This function adds a message to the message history from a JSON message
        :param json_message: A JSON message containing the message to add
        """

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
        """
        Clears the entire message history
        """
        self.history = []

    def clear_all_but_system(self):
        """
        Clears all messages except the system message
        """
        self.history = [self.history[0]]

    def get_history(self):
        """
        This function returns the underlying message history list
        """
        return self.history

    def last(self) -> str:
        """
        This function returns the content of the last message in the history
        """
        if self.history is None or len(self.history) == 0:
            raise ValueError("Message history is empty.")

        message_dict = self.history[-1]

        if "content" not in message_dict:
            raise ValueError("Message does not contain 'content' field.")

        return message_dict["content"]

    def get_last_message_role(self) -> str:
        """
        This function returns the role of the last message in the history
        """
        if self.history is None or len(self.history) == 0:
            raise ValueError("Message history is empty.")

        message_dict = self.history[-1]

        if "role" not in message_dict:
            raise ValueError("Message does not contain 'role' field.")

        return message_dict["role"]

    def chat_str(self) -> str:
        """
        This function returns the message history in the following format:
        SYSTEM: ...
        USER: ...
        ASSISTANT: ...
        ...
        """
        chat_str = ""

        for message in self.history:
            role = message["role"].upper()
            content = message["content"]
            chat_str += f"{role}: {content}" + "\n"

        return chat_str

    def __str__(self):
        return json.dumps(self.history, indent=2)


class LLMClient:
    """
    This class represents a client for interacting with a Large Language Model (LLM) API.

    :param url: The URL of the LLM API
    :param model: The model name to call for in the LLM API
    """

    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def invoke(self, message_history: MessageHistory) -> MessageHistory:
        """
        This function calls the LLM API with the given message history and returns
        the updated message history, with the LLM's response appended to it.
        """

        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "messages": message_history.get_history()}
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            message_history.add_message_from_json(json.dumps(response_data))
        else:
            response.raise_for_status()

        return message_history


def main():
    # noinspection PyUnresolvedReferences
    import readline

    llm_client = LLMClient(
        url="http://localhost:1234/v1/chat/completions",
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    )

    history = MessageHistory()
    history.add(
        "system", "You are a helpful assistant playing the game Minecraft."
    )

    while True:
        message = input("USER: ")
        history.add("user", message)

        history = llm_client.invoke(history)

        role = history.get_last_message_role()
        msg = history.last()
        print(f"{role.upper()}: {msg}")


if __name__ == "__main__":
    main()
