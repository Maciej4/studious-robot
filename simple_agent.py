"""
The simple agent is a refined version of the originally used approach.
The agent writes a plan, then executes it step by step.
"""
import time

from termcolor import colored

from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory
import mine_tools


class SimpleAgent(ThreadedAgent):
    # Write a plan, then execute it step by step
    def __init__(self, message_bus: MessageBus):
        super().__init__("simple", message_bus, self.observe_surroundings)
        self.llm: LLMClient = self.message_bus.get_resource("llm")
        self.plan = None
        self.history: MessageHistory = MessageHistory()

        system_prompt = (
                "You are an expert agent playing Minecraft. Given a task, "
                "perform it using the below tools, formatted as python functions:\n\n"
                + mine_tools.get_tools_string()
        )
        self.history.add("system", system_prompt)

        goal = (
            "Mine a log."
        )

        self.history.add("system", goal)

        print(colored(self.history, "yellow"))

    def observe_surroundings(self):
        self.history.add("user", "Here is a description of the starting surroundings:")
        print("USER: " + self.history.last())
        description = mine_tools.visual_question("Describe the surroundings.")
        self.history.add("user", description)
        print("MOLMO: " + self.history.last())
        return self.write_plan

    def write_plan(self):
        self.history.add("user", "Write a short plan to achieve the goal given the current surroundings.")
        print("USER: " + self.history.last())
        self.history = self.llm.invoke(self.history)
        print(colored("AGENT: " + self.history.last(), "cyan"))
        self.history.add("user",
                         "Execute the above plan step by step. Respond only with a single tool function call. Do not offer any further explanation.")
        print("USER: " + self.history.last())
        return self.execute_plan

    def execute_plan(self):
        self.history = self.llm.invoke(self.history)
        print(colored("AGENT: " + self.history.last(), "cyan"))

        if "mission complete" in self.history.last().lower():
            return self.done

        tool_call_result = mine_tools.exec_tool_call(self.history.last())
        self.history.add("user", tool_call_result)
        print("TOOLS: " + self.history.last())
        return self.execute_plan

    def done(self):
        print(colored("MISSION COMPLETE", "green"))
        exit()


def main():
    bus = MessageBus()

    llm = LLMClient(
        url="http://localhost:1234/v1",
        model="lmstudio-community/Llama-3.1-Tulu-3-8B-GGUF",
    )

    bus.add_resource("llm", llm)

    SimpleAgent(bus)

    bus.start_agents()

    while bus.running:
        time.sleep(1)

    bus.stop_agents()


if __name__ == "__main__":
    main()
