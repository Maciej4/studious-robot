from flask import Flask, request, jsonify
from llm_client import (
    MessageHistory,
    LLMClient,
)

app = Flask(__name__)


# Dummy chatbot function for demonstration purposes
def chatbot(messages: str) -> str:
    """
    Given the messages history in the format (not json; a single string)
    USER: message
    ASSISTANT: message
    USER: message
    ASSISTANT:

    the code will generate the next message from the assistant
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
    message_str = ""
    for message in history.get_history():
        message_str += f"{message['role'].upper()}: {message['content']}\n"

    # Get the assistant's response
    assistant_response = chatbot(message_str)

    # Add the assistant's response to the message history
    history.add_message("assistant", assistant_response)

    # Return the updated message history
    return jsonify(
        {
            "id": "chatcmpl-unique-id",  # You can generate a unique ID here
            "object": "chat.completion",
            "created": 1729806952,  # You can use the current timestamp here
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(message_str.split()),  # Example token count
                "completion_tokens": len(
                    assistant_response.split()
                ),  # Example token count
                "total_tokens": len(message_str.split())
                + len(assistant_response.split()),  # Example token count
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234)
