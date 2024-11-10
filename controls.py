import pyautogui
import time
import re

from llm_client import LLMClient, MessageHistory
from point_tracker import PointTracker


class MinecraftController:
    def __init__(self):
        self.llm_client = LLMClient(
            url="http://localhost:1234/v1/chat/completions",
            # The model name doesn't matter when it is hosted with host_model.py
            model="allenai/Molmo-7B-D",
        )

    def take_screenshot(self):
        # press f2 to take a screenshot
        pyautogui.press('f2')

        # press f3 and d at the same time to clear the chat message
        pyautogui.keyDown('f3')
        pyautogui.keyDown('d')
        pyautogui.keyUp('f3')
        pyautogui.keyUp('d')

        time.sleep(1)

    def switch_to_minecraft(self):
        pyautogui.moveTo(2890, 160, duration=0.2)
        time.sleep(0.1)
        pyautogui.click()
        time.sleep(0.1)
        pyautogui.press('escape')
        time.sleep(0.1)

    def point_to_pixels(self, message, width=1918, height=1016) -> (int, int):
        coordinates = re.findall(r'x1?="(.+?)" y1?="(.+?)"', message)

        if len(coordinates) == 0:
            return 0, 0

        point_raw_x = float(coordinates[0][0])
        point_raw_y = float(coordinates[0][1])

        dx = round((point_raw_x / 100) * width)
        dy = round((point_raw_y / 100) * height)

        return int(dx), int(dy)

    def look_at_point(self, msg) -> str:
        x, y = self.point_to_pixels(msg, width=640, height=480)

        if x == 0 and y == 0:
            print("No point found in input message")
            return "No point found"

        tracker = PointTracker((x, y), headless=False)

        time.sleep(2)

        success = False

        try:
            i = 0
            while tracker.running:
                tx, ty = tracker.get_latest_position()

                print(f"tx: {tx}, ty: {ty}")

                dx, dy = 320 - tx, 240 - ty
                coeff = 0.3

                if abs(dx) < 8 and abs(dy) < 8:
                    print("Reached target")
                    success = True
                    break

                dx, dy = round(-dx * coeff), round(-dy * coeff)

                pyautogui.move(dx, dy, duration=0.1)

                i += 1
                if i > 20:
                    break
        except KeyboardInterrupt:
            pass

        tracker.stop()
        tracker.thread.join()
        print("Stopped tracking")

        if success:
            return "Successfully looked at point"
        else:
            return "Failed to look at point"

    def look_at(self, object: str):
        """
        Call the vision model to identify the object, then look at it using the look_at_point method.
        """
        pyautogui.press('f1')

        self.take_screenshot()

        history = MessageHistory()
        look_at_prompt = """You are a helpful assistant playing the game Minecraft. Given an object,\
        you need to point it out. Only point out one instance of the object, even if there are multiple.\
        Prefer the nearest instance unless otherwise specified."""

        history.add_message("system", look_at_prompt)
        history.add_message("user", f"Point out the following: {object}.")

        history = self.llm_client.invoke(history)

        print(history)

        role = history.get_last_message_role()
        assert role == "assistant", "Expected assistant role"

        msg = history.get_last_message_str()
        print(f"{role.upper()}: {msg}")

        result = self.look_at_point(msg)

        pyautogui.press('f1')

        return result

    def block_distance_to_time(self, distance: float) -> float:
        return distance / 4.317

    def move_forward(self, distance: float):
        time.sleep(0.1)
        pyautogui.keyDown('w')
        time.sleep(self.block_distance_to_time(distance))
        pyautogui.keyUp('w')
        time.sleep(0.1)
        return "Moved forward"

    def mine_block(self):
        pyautogui.mouseDown()
        time.sleep(4)
        pyautogui.mouseUp()
        time.sleep(0.1)
        return "Block mined"

    def inventory_contains(self, item: str):
        self.take_screenshot()

        history = MessageHistory()
        inventory_contains_prompt = """You are a helpful assistant playing the game Minecraft.\
    Given an item, you need to check if it is in the player's hotbar or inventory.\
    Respond only with a yes or no answer.\
    The hotbar is the row of items or empty slots at the bottom of the screen."""

        history.add_message("system", inventory_contains_prompt)
        history.add_message("user", f"Does the player have {item} in their inventory?")

        history = self.llm_client.invoke(history)

        role = history.get_last_message_role()
        assert role == "assistant", "Expected assistant role"

        return history.get_last_message_str()

    def visual_question(self, question: str):
        """
        Call the vision model to answer the question.
        """
        self.take_screenshot()

        history = MessageHistory()
        visual_question_prompt = """You are a helpful assistant playing the game Minecraft.\
        Provide short but detailed answers the the questions you are given.\
        Include the distances to all objects in the scene."""

        history.add_message("system", visual_question_prompt)
        history.add_message("user", question)

        history = self.llm_client.invoke(history)

        role = history.get_last_message_role()
        assert role == "assistant", "Expected assistant role"

        return history.get_last_message_str()

    def main(self):
        while True:
            history = MessageHistory()
            history.add_message("system", "You are a helpful assistant playing the game Minecraft.")

            message = input("USER: ")

            if message == "exit":
                break

            if message == "":
                message = "Point out the center of the bullseye surrounded by blue."

            history.add_message("user", message)

            history = self.llm_client.invoke(history)

            role = history.get_last_message_role()
            msg = history.get_last_message_str()
            print(f"{role.upper()}: {msg}")

            self.switch_to_minecraft()

            self.look_at_point(msg)


"""
Serve the tools as an API
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

controller = MinecraftController()


@app.route('/look_at', methods=['POST'])
def api_look_at():
    data = request.get_json()
    object_to_look_at = data['object']

    print("Looking at:", object_to_look_at)

    result = controller.look_at(object_to_look_at)
    print("Result:", result)

    return jsonify(result=result)


@app.route('/move_forward', methods=['POST'])
def api_move_forward():
    data = request.json
    controller.move_forward(data['distance'])
    return jsonify(result="Moved forward")


@app.route('/mine_block', methods=['POST'])
def api_mine_block():
    controller.mine_block()
    return jsonify(result="Block mined")


@app.route('/visual_question', methods=['POST'])
def api_visual_question():
    data = request.get_json()
    question = data['question']
    print("Question:", question)

    result = controller.visual_question(question)
    print("Result:", result)

    return jsonify(result=result)


@app.route('/inventory_contains', methods=['POST'])
def api_inventory_contains():
    data = request.json
    result = controller.inventory_contains(data['item'])
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4321, debug=True)
