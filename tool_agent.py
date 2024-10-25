from llm_graph import LLMGraph
from llm_tools import (
    tool,
    are_tools_present,
    extract_and_run_tools,
    tools_to_string,
)
from llm_client import LLMClient, MessageHistory


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b


tools = [add, subtract, multiply, divide]

llm = LLMClient(
    url="http://localhost:1234/v1/chat/completions",
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
)


# Define nodes for the graph
def chatbot(messages: MessageHistory) -> MessageHistory:
    # print(messages)
    response = llm.invoke(messages)
    return response


def route_tools(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    if are_tools_present(last_message):
        return "tools"
    return "END"


def tool_node(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    tool_results = extract_and_run_tools(last_message, tools)
    # tool_result_message = "\n".join(tool_results)
    tool_results = f"```\n{tool_results}\n```\n Tools have been executed. Have you completed your task?"
    messages.add_message("user", tool_results)
    return messages


# Create the graph
graph = LLMGraph()
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(graph.START, "chatbot")
graph.add_edge("tools", "chatbot")
graph.add_conditional_edge("chatbot", route_tools, {"tools": "tools", "END": "END"})

tools_str = tools_to_string(tools)
system_prompt = """You are a helpful assistant tasked with performing calculations.\
Provide concise but detailed responses to the user's queries.\

You have been given access to a set of tools to help you perform your task efficiently.\
These tools are python function, which can be used by placing the function with its\
arguments in a python code block at the end of your response. The functions are already\
pre-defined, so do not attempt to redefine them. Do not define functions. Do not use\
functions that are not provided. Do not use loops, conditionals, or any other python\
constructs. The python code block must include three backticks and the word python on\
the line above and three backticks on the line below. The code block is in markdown format.\
Do not use nested functions. You do not need to use print statements, as the functions\
already print the result.

For example, to add 2 and 3, you would write:

```python
add(2, 3)
```

The tools available to you are:
"""

system_prompt += tools_str

print("SYSTEM:", system_prompt)
print()


# Function to stream graph updates
def stream_graph_updates(user_input):
    initial_state = MessageHistory()
    initial_state.add_message("system", system_prompt)
    initial_state.add_message("user", user_input)
    for state in graph.stream(initial_state):
        # print("a")
        print()
        print(state.get_last_message_role().upper() + ":", state.get_last_message_str())
        # print(state)
        # for message in state["messages"]:
        #     print(f"{message['role'].capitalize()}: {message['content']}")


# Main loop
while True:
    user_input = input("USER: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
