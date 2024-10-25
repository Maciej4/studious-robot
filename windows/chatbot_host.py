from flask import Flask, request, jsonify
from llm_client import (
    MessageHistory,
)


app = Flask(__name__)


# Dummy chatbot function for demonstration purposes
def chatbot(messages: str) -> str:
    """
    This function should take a string of messages in the following format:
    USER: message
    ASSISTANT: message
    ...
    """
    # This is a placeholder implementation. Replace it with your actual logic.
    return "This is the assistant's response."


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    if not data or "model" not in data or "messages" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    model = data["model"]
    messages = data["messages"]

    # Initialize message history
    history = MessageHistory()
    for message in messages:
        history.add_message(message["role"], message["content"])

    # Convert message history to the required string format for the chatbot function
    message_str = history.pretty_print()

    # Invoke the LLM model
    assistant_response = chatbot(message_str)

    history.add_message("assistant", assistant_response)

    return jsonify(
        {
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_response},
                    "finish_reason": "stop",
                }
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234)
