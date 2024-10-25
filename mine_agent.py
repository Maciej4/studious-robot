from llm_client import LLMClient, MessageHistory
from llm_graph import LLMGraph
from llm_tools import (
    are_tools_present,
    extract_and_run_tools,
    tool,
    tools_to_string,
)


@tool
def look_at(object: str) -> str:
    """Given an object, look at it."""
    out = input(f"Look at {object}? ")
    return out if out != "" else f"Success, currently looking at {object}."


@tool
def move_forward(distance: int) -> str:
    """
    Move forward a specified number of blocks. Note that moving
    forward means moving in the direction you are looking. It is
    recommended to look at the object you want to move towards
    before using this tool.
    """
    out = input(f"Move forward {distance} blocks? ")
    return out if out != "" else f"Success, moved forward {distance} blocks."


@tool
def mine_block() -> str:
    out = input("Mine the block? ")
    """
    Mine the block that you are looking at. This only works if
    the block is within 3 blocks of the player and the player is
    looking at it.
    """
    return out if out != "" else "Success, block mined."


@tool
def visual_question(question: str) -> str:
    """
    The vision model will provide a response to the question.
    This is very broad and can range from describing the scene
    in general, to identifying objects, to answering specific
    detailed questions about the scene or the player.
    """
    return input(f"Ask the vision model: {question}? ")


@tool
def inventory_contains(item: str) -> str:
    """
    Check if the player has a particular item in their inventory.
    One use for this is checking if a mined block has been
    successfully picked up and added to the inventory.
    """
    out = input(f"Does the inventory contain {item}? ")
    return out if out != "" else f"Yes, your inventory contains {item}."


tools = [look_at, move_forward, mine_block, visual_question, inventory_contains]

llm = LLMClient(
    url="http://localhost:1234/v1/chat/completions",
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
)


# Define nodes for the graph
def chatbot(messages: MessageHistory) -> MessageHistory:
    response = llm.invoke(messages)
    return response


def route_tools(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    if are_tools_present(last_message):
        return "tools"
    return "END"


tool_use_message = "Tools have been executed. Have you completed your task?"


def tool_node(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    tool_results = extract_and_run_tools(last_message, tools)
    tool_results = f"```\n{tool_results}\n```\n {tool_use_message}"
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
system_prompt = """You are an AI agent playing Minecraft.\
Keep responses short but filled with details.\
You are working with another language model which has vision capabilities and can see the Minecraft game.\
This model can describe the scene and point out objects, while you need to plan the actions\
to take based on these descriptions in order to achieve a certain goal.\
Describe your thought process, then output a python code block (three backticks, python, code, three backticks)\
with the singular aciton you want to take.

IMPORTANT: Only provide a single action at a time and provide this action at the end of your response.\
Do not assume the role of the vision model.

HINTS: Make use of broader questions like:

```python
visual_question("Describe the scene.")
```

to get a general idea of the scene before asking more specific questions.

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
        print()
        print(state.get_last_message_role().upper() + ":", state.get_last_message_str())


while True:
    user_input = input("USER: ")

    if user_input == "":
        user_input = "GOAL: Mine 1 log from a tree."

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
