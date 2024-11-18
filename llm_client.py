import requests
import json

from agent_base import Message


class MessageHistory:
    """
    This class represents a conversation history in the form of a list of messages.

    It includes transaction management, message retrieval, and message appending.
    """

    def __init__(self):
        self.history = []
        self.history_stack = []

    def start_transaction(self):
        """
        This function starts a transaction by saving the current history to a stack.
        """
        self.history_stack.append(self.history.copy())

    def rollback(self):
        """
        This function rolls back the history to the last transaction.
        """
        if not self.history_stack:
            raise ValueError("No transaction to rollback.")

        self.history = self.history_stack.pop()

    def commit(self):
        """
        This function commits the current history.
        """
        if not self.history_stack:
            raise ValueError("No transaction to commit.")

        self.history_stack.clear()

    def add(self, role: str, content: str):
        """
        This function adds a message to the history.
        """
        self.history.append(Message(role, content))

    def clear_all_but_system(self):
        """
        Clears all messages except the system message
        """
        self.history = [self.history[0]]

    def to_api_format(self) -> list[dict]:
        """
        This function returns the JSON-like representation of the message history.
        It also merges repeated user and assistant messages to comply with the
        expected API format.
        """

        assert self.history, "No messages in the history."

        new_history = []
        current_new_message = Message("system", "")

        # Merge repeated user and assistant messages with \n as separator
        for message in self.history:
            if message.role == current_new_message.role:
                current_new_message.content += "\n" + message.content
            else:
                new_history.append(current_new_message)
                current_new_message = message.copy()

        new_history.append(current_new_message)

        assert new_history[0].role == "system", "First message must be a system message."
        assert new_history[-1].role == "user", "Last message must be a user message."

        for i in range(1, len(new_history) - 1):
            if i % 2 == 1:
                assert new_history[i].role == "user", "User message expected."
            else:
                assert new_history[i].role == "assistant", "Assistant message expected."

        # Marshal the history to JSON
        # return json.dumps([message.to_json() for message in new_history])
        return [message.to_dict() for message in new_history]

    def add_from_json(self, json_message_str: str):
        """
        This function appends messages from a JSON string to the history.
        It is typically used to append API responses to the history.
        """

        try:
            json_message = json.loads(json_message_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format.")

        if "choices" not in json_message or len(json_message["choices"]) == 0:
            raise ValueError("JSON message must contain 'choices' with at least one valid message.")

        for choice in json_message["choices"]:
            if ("message" not in choice
                    or "role" not in choice["message"]
                    or "content" not in choice["message"]):
                raise ValueError("Invalid message format.")

            self.add(choice["message"]["role"], choice["message"]["content"])

    def last(self) -> str:
        """
        This function returns the content of the last message in the history.
        """
        if not self.history:
            raise ValueError("Message history is empty.")

        return self.history[-1].content

    def last_role(self) -> str:
        """
        This function returns the role of the last message in the history.
        """
        if not self.history:
            raise ValueError("Message history is empty.")

        return self.history[-1].role

    def __str__(self):
        return "\n".join([str(message) for message in self.history])


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
        payload = {"model": self.model, "messages": message_history.to_api_format()}
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            message_history.add_from_json(json.dumps(response_data))
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

        role = history.last_role()
        msg = history.last()
        print(f"{role.upper()}: {msg}")


if __name__ == "__main__":
    main()
