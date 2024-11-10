from llm_client import LLMClient, MessageHistory
from llm_graph import LLMGraph
from llm_tools import (
    are_tools_present,
    extract_and_run_tools,
    tool,
    tools_to_string,
)
import json
import requests


try:
    with open("env.json") as f:
        env = json.load(f)
        controls_base_url = env["controls_base_url"]
except FileNotFoundError:
    exception_message = (
        "Create a file called env.json with the key controls_base_url. "
        "This base url should point to the address where the controls API is hosted. "
        "For example, http://localhost:5000"
    )

    raise FileNotFoundError(exception_message)

headers = {"Content-Type": "application/json"}


def call_controller_api(endpoint: str, payload: dict) -> dict:
    response = requests.post(
        f"{controls_base_url}/{endpoint}", headers=headers, data=json.dumps(payload)
    )

    assert response.status_code == 200, f"Error: {response.text}"
    return response.json()["result"]


@tool
def look_at(object: str) -> str:
    """Given an object, look at it."""
    # out = input(f"Look at {object}? ")
    # return out if out != "" else f"Success, currently looking at {object}."

    print(f"Looking at {object}")
    response_string = call_controller_api("look_at", {"object": object})
    print(response_string)

    return response_string


@tool
def move_forward(distance: int) -> str:
    """
    Move forward a specified number of blocks. The direction to
    move in should be specified by using the `look_at` tool before
    using this tool. Small increments of 5 blocks or less are
    recommended to avoid collisions with obstacles.
    """
    # out = input(f"Move forward {distance} blocks? ")
    # return out if out != "" else f"Success, moved forward {distance} blocks."
    print(f"Moving forward {distance} blocks")
    response_string = call_controller_api("move_forward", {"distance": distance})
    print(response_string)

    return response_string


@tool
def mine_block() -> str:
    """
    Mine the block that you are looking at. This only works if
    the block is within 3 blocks of the player and the player is
    looking at it. Confirm the distance before mining the block.
    The vision model should mention a purple outline around the
    block if it is mineable. Check your inventory to see if the
    block has been added. If not, move forward and try again.
    """
    # return out if out != "" else "Success, block mined."

    print("Mining the block")
    response_string = call_controller_api("mine_block", {})
    print(response_string)

    # response_string = input("Was mining successful? ")

    return "Block may have been mined. Ensure that the block is within 3 blocks of the player and the player is looking at it."


@tool
def visual_question(question: str) -> str:
    """
    The vision model will provide a response to the question.
    This is very broad and can range from describing the scene
    in general, to identifying objects, to answering specific
    detailed questions about the scene or the player.
    """
    # Call the visual_question endpoint at the controls_base_url
    # with the question as a query parameter using a POST request

    print(f"Question: {question}")
    response_string = call_controller_api("visual_question", {"question": question})
    print(response_string)

    return response_string


@tool
def inventory_contains(item: str) -> str:
    """
    Check if the player has a particular item in their inventory.
    One use for this is checking if a mined block has been
    successfully picked up and added to the inventory.
    """
    # out = input(f"Does the inventory contain {item}? ")
    # return out if out != "" else f"Yes, your inventory contains {item}."

    print(f"Checking if the inventory contains {item}")
    response_string = call_controller_api("inventory_contains", {"item": item})
    print(response_string)

    return response_string


tools = [look_at, move_forward, mine_block, visual_question, inventory_contains]

llm = LLMClient(
    url="http://localhost:1234/v1/chat/completions",
    model="lmstudio-community/qwen2.5-14b-instruct",
    # model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
)


mastermind_prompt = """You are an AI agent playing Minecraft.\
You are the mastermind of the operation and work with a team of other language models to achive\
a user-defined goal. Given a goal, split it into a step-by-step plan to be executed by the team.\
Describe all steps, such as gather information, looking at objects, moving, mining, etc.\
However, you cannot execute the steps yourself, you can only plan them.\
"""


def mastermind(messages: MessageHistory):
    """
    The mastermind function is the main function, which is responsible for
    creating the overall plan for the agent to achieve its goal.
    """
    mastermind_message = MessageHistory()
    mastermind_message.add_message("system", mastermind_prompt)

    goal = messages.get_last_message_str()

    mastermind_message.add_message("user", goal)
    response = llm.invoke(mastermind_message)

    messages.add_message("user", response.get_last_message_str())

    return messages


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
graph.add_node("mastermind", mastermind)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
# graph.add_edge(graph.START, "chatbot")
graph.add_edge(graph.START, "mastermind")
graph.add_edge("mastermind", "chatbot")
graph.add_edge("tools", "chatbot")
graph.add_conditional_edge("chatbot", route_tools, {"tools": "tools", "END": "END"})

tools_str = tools_to_string(tools)
# system_prompt = """You are an AI agent playing Minecraft.\
# You are the member of a team of language models working together to achieve a goal.\
# The mastermind model has created a plan above to achive the goal, you
# need to execute it step by step.\
# You are working with another language model which has vision capabilities and can see the Minecraft game.\
# This model can describe the scene and point out objects, while you need to plan the actions\
# to take based on these descriptions in order to achieve a certain goal.\

# Don't write a plan, just execute the steps of the plan one at a time.\
# The tools you can use are:
# """

system_prompt = """You are an AI agent playing Minecraft.\
You are working with a vision language model to execute the above plan.\
You need to execute the steps of the plan one at a time and\
correct any errors that occur. Mention what step of the plan you\
are on and what you are doing. Assume that you start on step 1.\

At the end of each response, include a tool to use next inside\
triple backticks (```), for example:

```
visual_question("Describe the scene.")
```

Do not include `object=`, `tool=`, or `action=` in the tool call.\
Just the function name and the arguments.

Do not guess the output of functions or speculate about the game state.\
Assume that you start with no items.\
You can mine a log with your hands.

The tools you can use are:
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
