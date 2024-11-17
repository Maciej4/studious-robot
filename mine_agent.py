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
def turn(direction: str) -> str:
    """
    Turn in a specified direction.
    Constraints:
    - The direction must be one of the following: 'left', 'right', 'up', or 'down'.
    - Rotations are in 45-degree increments.
    """
    response_string = call_controller_api("turn", {"direction": direction})

    return response_string


@tool
def look_at(object: str) -> str:
    """
    Given an object, look at it.
    Constraints:
    - The object must be within the player's field of view.
    """
    # out = input(f"Look at {object}? ")
    # return out if out != "" else f"Success, currently looking at {object}."

    # print(f"Looking at {object}")
    response_string = call_controller_api("look_at", {"object": object})
    # print(response_string)

    return response_string


@tool
def move_forward(distance: int) -> str:
    """
    Move forward a specified number of blocks.
    Constraints:
    - You are looking at the object you want to move towards (use the `look_at` tool).
    - Make sure the path is clear before moving forward larger distances.
    - Do not move forward more than 100 blocks at a time.
    """
    # out = input(f"Move forward {distance} blocks? ")
    # return out if out != "" else f"Success, moved forward {distance} blocks."
    # print(f"Moving forward {distance} blocks")
    response_string = call_controller_api("move_forward", {"distance": distance})
    # print(response_string)

    return response_string


@tool
def mine_block() -> str:
    """
    Mine the block that you are looking at.
    Constraints:
    - The block must be within 3 blocks of the player. Check this by asking if there is a purple outline around the block.
    - The player must be looking at the block.
    - Move forward after mining the block to pick up the dropped item.
    - Success can be checked by using the `inventory_contains` tool.
    """
    # return out if out != "" else "Success, block mined."

    # print("Mining the block")
    response_string = call_controller_api("mine_block", {})
    # print(response_string)

    # response_string = input("Was mining successful? ")

    return "Block may have been mined. Ensure that the block is within 3 blocks of the player and the player is looking at it."


@tool
def visual_question(question: str) -> str:
    """
    The vision model will provide a response to the question.
    This is very broad and can range from describing the scene
    in general, to identifying objects, to answering specific
    detailed questions about the scene or the player. Do not
    ask the vision model about how to proceed or what the state
    of the inventory is, use the `replan` and `inventory_contains`
    tools for those purposes respectively.
    """
    # Call the visual_question endpoint at the controls_base_url
    # with the question as a query parameter using a POST request

    # print(f"Question: {question}")
    response_string = call_controller_api("visual_question", {"question": question})
    # print(response_string)

    return response_string


@tool
def inventory_contains() -> str:
    """
    Lists the items in the player's inventory.
    """
    # out = input(f"Does the inventory contain {item}? ")
    # return out if out != "" else f"Yes, your inventory contains {item}."

    item = "null"

    # print(f"Checking if the inventory contains {item}")
    response_string = call_controller_api("inventory_contains", {"item": item})
    # print(response_string)

    return response_string


@tool
def replan(history: str) -> str:
    """
    Ask the Mastermind for insights into your current situation.
    Useful for getting unstuck or when the plan needs to be updated
    dramatically. The history parameter should contain a summary of
    what you have done so far and what your issue or question is.
    """

    replan_messages = MessageHistory()

    replan_system_message = (
            "You are an expert at resolving issues and replanning tasks for a Minecraft agent. Provide guidance to the agent on how to proceed next and look for potential flaws in what the agent has done so far. Particularily focus on tool usage and ensure tools are used correctly.\n\nThe agent has the following tools:"
            + tools_str
    )

    replan_messages.add_message("system", replan_system_message)
    replan_messages.add_message("user", history)
    replan_response = llm.invoke(replan_messages)

    return replan_response.last()


@tool
def ask_user(question: str) -> str:
    """
    Ask the operator a question. Only use this tool in emergencies.
    Or if you have completed your task and need further instructions.
    """
    return input(question + " ")


@tool
def done() -> str:
    """
    Once you have completed the task, use this tool to signal the Mastermind that you are done.
    """
    return "Have you completed your task? Answer only with 'yes' or 'no'. Do not provide any other information."


tools = [
    look_at,
    move_forward,
    mine_block,
    visual_question,
    inventory_contains,
    replan,
    ask_user,
    done,
]

llm = LLMClient(
    url="http://localhost:1234/v1/chat/completions",
    model="arcee-ai/SuperNova-Medius-GGUF",
    # model="lmstudio-community/qwen2.5-14b-instruct",
    # model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
)

mastermind_prompt = "You are an AI agent playing Minecraft. You are the mastermind of the operation and work with a team of other language models to achive a user-defined goal. Given a goal, split it into a step-by-step plan to be executed by the team. Describe all steps, such as gather information, looking at objects, moving, mining, etc. However, you cannot execute the steps yourself, you can only plan them. For each step, include constraints that need to be met before the step can be executed. Keep your plan simple, short, and clear. Assume you start with nothing."


def mastermind(messages: MessageHistory):
    """
    The mastermind function is the main function, which is responsible for creating the overall plan for the agent to achieve its goal.
    """
    mastermind_message = MessageHistory()
    mastermind_message.add_message("system", mastermind_prompt)

    goal = messages.last()

    mastermind_message.add_message("user", goal)
    response = llm.invoke(mastermind_message)

    messages.add_message("user", response.last())

    return messages


# Define nodes for the graph
def chatbot(messages: MessageHistory) -> MessageHistory:
    response = llm.invoke(messages)
    return response


def route_tools(messages: MessageHistory):
    last_message = messages.last()
    if are_tools_present(last_message):
        return "tools"
    return "END"


tool_use_message = "Tools have been executed. Have you completed your task?"


def tool_node(messages: MessageHistory):
    last_message = messages.last()
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

# system_prompt = """You are an AI agent playing Minecraft. You are working with a vision language model to execute the above plan. You need to execute the steps of the plan one step at a time and correct any errors that occur. Mention what step of the plan you are on and what you are doing. Start at the first step of the plan.

# Include a single function call in each step and keep your responses short. Do not include `object=`, `tool=`, or `action=` in the tool call. Just the function name and the arguments.

# Before each tool call, list the associated constraints. If they are satisfied, describe what tells you they are satisfied. If they are not satisfied, describe what you need to do to satisfy them and don't execute the tool call.

# Do not guess the output of functions or speculate about the game state. Assume that you start with no items. You can mine a log with your hands.

# The tools you can use are:
# """

system_prompt = """You are an AI agent playing Minecraft. You are working with a vision language model to execute a plan.

Execute the steps of the plan one by one starting at the first step. At each step, restate the step and list all constraints both for the step and the tools you want to use. Respond only with a single tool call that executes the next step of the plan. Do not describe your though process or speculate on the game state.

Include a single function call in each step and keep your responses short. Do not include `object=`, `tool=`, or `action=` in the tool call. Just the function name and the arguments.

The tools you can use are:
"""

tools_str = tools_to_string(tools)
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
        print(state.get_last_message_role().upper() + ":", state.last())


while True:
    user_input = input("USER: ")

    if user_input == "":
        user_input = "GOAL: Mine 1 log from a tree."

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
