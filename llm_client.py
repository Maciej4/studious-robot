import time
import json

from openai import OpenAI
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
        if len(self.history) > 1:
            self.history = [self.history[0]]

    def view(self, agent: str) -> list[Message]:
        """
        This function returns the message history from the perspective of the agent.
        It also merges repeated user and assistant messages to comply with the
        expected API format.

        :param agent: Which agent's perspective to use for the conversation.
            Messages from the agent's perspective will be renamed to "assistant".
            Other agents' messages will be renamed to "user". System messages of
            the form "system_<agent>" will be renamed to "system" with the other
            agents' system messages removed. "system" messages will be kept as is.
        """

        if not self.history:
            raise ValueError("Message history is empty.")

        # Rename all messages where the role matches the agent to "assistant"
        # since "assistant" represents the responses of the agent in the API.
        # Also, remove the system messages for other agents.
        if agent:
            renamed_history = MessageHistory()

            for message in self.history:
                if message.role == agent:
                    renamed_history.add("assistant", message.content)
                elif message.role == "system" or message.role == "system_" + agent:
                    renamed_history.add("system", message.content)
                elif not message.role.startswith("system_"):
                    new_content = message.role.upper() + ": " + message.content
                    renamed_history.add("user", new_content)

            renamed_history = renamed_history.history
        else:
            renamed_history = self.history.copy()

        new_history = []
        current_new_message = Message("", "")

        # Merge repeated user and assistant messages with \n as separator
        for message in renamed_history:
            if not agent and message.role not in ["system", "user", "assistant"]:
                raise ValueError("Invalid role in message. Pass in agent to merge messages.")

            if message.role == current_new_message.role:
                current_new_message.content += "\n" + message.content
            else:
                if current_new_message.role:
                    new_history.append(current_new_message)
                current_new_message = message.copy()

        new_history.append(current_new_message)

        if not new_history[0].role == "system":
            raise ValueError("First message must be a system message.")

        if not new_history[-1].role == "user":
            raise ValueError("Last message must be a user message.")

        for i in range(1, len(new_history) - 1):
            if i % 2 == 1 and new_history[i].role != "user":
                raise ValueError("Incorrect ordering: User message expected.")
            elif i % 2 == 0 and new_history[i].role != "assistant":
                raise ValueError("Incorrect ordering: Assistant message expected.")

        return new_history

    def to_api_format(self, agent=None) -> list[dict]:
        """
        This function returns the JSON-like representation of the message history.
        It also merges repeated user and assistant messages to comply with the
        expected API format.

        :param agent: Which agent's perspective to use for the conversation.
            Messages from the agent's perspective will be renamed to "assistant".
            Other agents' messages will be renamed to "user". System messages of
            the form "system_<agent>" will be renamed to "system" with the other
            agents' system messages removed. "system" messages will be kept as is.
        """

        new_history = self.view(agent)

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

    def set_last_role(self, role: str):
        """
        This function sets the role of the last message in the history.
        """
        if not self.history:
            raise ValueError("Message history is empty.")

        self.history[-1].role = role

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
        self.client = OpenAI(base_url=url, max_retries=100)

    def invoke(self, message_history: MessageHistory, agent=None, max_tokens=1024) -> MessageHistory:
        """
        This function calls the LLM API with the given message history and returns
        the updated message history, with the LLM's response appended to it.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_history.to_api_format(agent),
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        message_history.add("assistant", content)
        return message_history


def main():
    # noinspection PyUnresolvedReferences
    import readline

    llm_client = LLMClient(
        url="http://localhost:1234/v1",
        model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    )

    history = MessageHistory()
    history.add(
        "system", "You are a helpful assistant playing the game Minecraft."
    )

    while True:
        message = input("USER: ")
        history.add("user", "USER: " + message + "\nRead the question again: USER: " + message)

        start = time.time_ns()
        history = llm_client.invoke(history)
        end = time.time_ns()
        print(f"Assistant response time: {(end - start) / 1e6:.2f} ms")

        role = history.last_role()
        msg = history.last()
        print(f"{role.upper()}: {msg}")


if __name__ == "__main__":
    main()
