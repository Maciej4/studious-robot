from llm_tools import (
    are_tools_present,
    extract_and_run_tools,
    tool,
    tools_to_string,
)
import json
import requests


class MineTools:
    def __init__(self):
        self.controls_base_url = self.setup_controller_api()
        self.headers = {"Content-Type": "application/json"}

    @staticmethod
    def setup_controller_api():
        try:
            with open("env.json") as f:
                env = json.load(f)
                return env["controls_base_url"]
        except FileNotFoundError:
            exception_message = (
                "Create a file called env.json with the key controls_base_url. "
                "This base url should point to the address where the controls API is hosted. "
                "For example, http://localhost:5000"
            )

            raise FileNotFoundError(exception_message)

    def call_controller_api(self, endpoint: str, payload: dict) -> str:
        response = requests.post(
            f"{self.controls_base_url}/{endpoint}", headers=self.headers, data=json.dumps(payload)
        )

        assert response.status_code == 200, f"Error: {response.text}"
        return response.json()["result"]

    @tool
    def turn(self, direction: str) -> str:
        """
        Turn in a specified direction.
        Constraints:
        - The direction must be one of the following: 'left', 'right', 'up', or 'down'.
        - Rotations are in 45-degree increments.
        """
        return self.call_controller_api("turn", {"direction": direction})

    @tool
    def look_at(self, target: str) -> str:
        """
        Given an object, look at it.
        Constraints:
        - The object must be within the player's field of view.
        """
        return self.call_controller_api("look_at", {"object": target})

    @tool
    def move_forward(self, distance: int) -> str:
        """
        Move forward a specified number of blocks.
        Constraints:
        - You are looking at the object you want to move towards (use the `look_at` tool).
        - Make sure the path is clear before moving forward larger distances.
        - Do not move forward more than 100 blocks at a time.
        """
        return self.call_controller_api("move_forward", {"distance": distance})

    @tool
    def mine_block(self) -> str:
        """
        Mine the block that you are looking at.
        Constraints:
        - The block must be within 3 blocks of the player. Check this by asking if there is a purple outline around the block.
        - The player must be looking at the block.
        - Move forward after mining the block to pick up the dropped item.
        - Success can be checked by using the `inventory_contains` tool.
        """
        _ = self.call_controller_api("mine_block", {})
        return "Block may have been mined."

    @tool
    def visual_question(self, question: str) -> str:
        """
        The vision model will provide a response to the question.
        This is very broad and can range from describing the scene
        in general, to identifying objects, to answering specific
        detailed questions about the scene or the player. Do not
        ask the vision model about how to proceed or what the state
        of the inventory is, use the `replan` and `inventory_contains`
        tools for those purposes respectively.
        """
        return self.call_controller_api("visual_question", {"question": question})

    @tool
    def inventory_contains(self) -> str:
        """
        Lists the items in the player's inventory.
        """
        return self.call_controller_api("inventory_contains", {"item": "null"})

    @tool
    def replan(self, history: str) -> str:
        """
        Ask the Mastermind for insights into your current situation.
        Useful for getting unstuck or when the plan needs to be updated
        dramatically. The history parameter should contain a summary of
        what you have done so far and what your issue or question is.
        """
        raise NotImplementedError
        # return self.call_controller_api("replan", {"history": history})

    @tool
    def ask_user(self, question: str) -> str:
        """
        Ask the operator a question. Only use this tool in emergencies.
        Or if you have completed your task and need further instructions.
        """
        raise NotImplementedError
        # return input(question + " ")

    @tool
    def done(self) -> str:
        """
        Once you have completed the task, use this tool to signal the Mastermind that you are done.
        """
        return "Have you completed your task? Answer only with 'yes' or 'no'. Do not provide any other information."

    def get_tools(self):
        tools = [
            self.look_at,
            self.move_forward,
            self.mine_block,
            self.visual_question,
            self.inventory_contains,
            # self.replan,
            # self.ask_user,
            # self.done,
        ]

        return tools

    def get_tools_string(self):
        return tools_to_string(self.get_tools())

    def get_tool_call(self, message: str) -> str:
        tools = self.get_tools()
        if are_tools_present(message):
            return extract_and_run_tools(message, tools)
        return "No tools found in the message."
