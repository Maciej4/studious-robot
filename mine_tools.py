from llm_tools import (
    are_tools_present,
    extract_and_run_tools,
    tool,
    tools_to_string,
)
import json
import requests


# --------------------- Tools ---------------------

@tool
def turn(direction: str) -> str:
    """
    Turn in a specified direction.
    Constraints:
    - The direction must be one of the following: 'left', 'right', 'up', or 'down'.
    - Rotations are in 45-degree increments.
    """
    return call_controller_api("turn", {"direction": direction})


@tool
def look_at(target: str) -> str:
    """
    Given an object, look at it.
    Constraints:
    - The object must be within the player's field of view.
    """
    return call_controller_api("look_at", {"object": target})


@tool
def move_forward(distance: int) -> str:
    """
    Move forward a specified number of blocks.
    Constraints:
    - You are looking at the object you want to move towards (use the `look_at` tool).
    - Make sure the path is clear before moving forward larger distances.
    - Do not move forward more than 100 blocks at a time.
    """
    return call_controller_api("move_forward", {"distance": distance})


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
    _ = call_controller_api("mine_block", {})
    return "Block may have been mined."


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
    return call_controller_api("visual_question", {"question": question})


@tool
def inventory_contains() -> str:
    """
    Lists the items in the player's inventory.
    """
    response = call_controller_api("inventory_contains", {"item": "null"})

    if not response:
        return "Your inventory is empty."

    return response


@tool
def place_block() -> str:
    """
    Place the block that you are holding. Make sure to look at the ground or
    another valid location to place the block.
    """
    return call_controller_api("interact", {"item": "null"})


@tool
def craft_item(item: str) -> str:
    """
    Craft an item using the items in the player's inventory.
    """
    return call_controller_api("craft", {"item": item})


@tool
def replan(history: str) -> str:
    """
    Ask the Mastermind for insights into your current situation.
    Useful for getting unstuck or when the plan needs to be updated
    dramatically. The history parameter should contain a summary of
    what you have done so far and what your issue or question is.
    """
    raise NotImplementedError
    # return self.call_controller_api("replan", {"history": history})


@tool
def ask_user(question: str) -> str:
    """
    Ask the operator a question. Only use this tool in emergencies.
    Or if you have completed your task and need further instructions.
    """
    raise NotImplementedError
    # return input(question + " ")


@tool
def done() -> str:
    """
    Once you have completed the task, use this tool to signal the Mastermind that you are done.
    """
    return "Have you completed your task? Answer only with 'yes' or 'no'. Do not provide any other information."


# --------------------- Infrastructure ---------------------

def get_tools():
    tools = [
        look_at,
        move_forward,
        mine_block,
        visual_question,
        inventory_contains,
        # self.replan,
        # self.ask_user,
        # self.done,
        place_block,
        craft_item,
        turn,
    ]

    return tools


def get_tools_string():
    return tools_to_string(get_tools())


def exec_tool_call(message: str) -> str:
    tools = get_tools()
    if are_tools_present(message):
        return extract_and_run_tools(message, tools)
    return "No tools found in the message."


try:
    with open("env.json") as f:
        env = json.load(f)
        controls_base_url = env["controls_base_url"]
        headers = {"Content-Type": "application/json"}
except FileNotFoundError:
    exception_message = (
        "Create a file called env.json with the key controls_base_url. "
        "This base url should point to the address where the controls API is hosted. "
        "For example, http://localhost:5000"
    )

    raise FileNotFoundError(exception_message)


def call_controller_api(endpoint: str, payload: dict) -> str:
    response = requests.post(
        f"{controls_base_url}/{endpoint}", headers=headers, data=json.dumps(payload)
    )

    assert response.status_code == 200, f"Error: {response.text}"
    return response.json()["result"]
