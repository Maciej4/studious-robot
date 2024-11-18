import time
import logging
from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory
from mine_tools import MineTools

logging.basicConfig(level=logging.INFO)


class MotorControlAgent(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.waiting_for_plan)
        self.plan = None
        self.task_completed = False
        self.llm = self.message_bus.get_resource("llm")
        self.history = MessageHistory()
        self.tools = MineTools()

        system = (
            "You are an AI agent playing Minecraft. "
            "Given a plan to perform a task, you need to execute the plan step by step. "
            # "\nThe tools you can use are listed below.\n"
            # + self.tools.get_tools_string()
        )

        print(system)

        self.history.add("system", system)

    def waiting_for_plan(self):
        message = self.receive_message(timeout=1)

        if message:
            self.plan = message.content
            # logging.info(f"{self.name}: Received plan '{self.plan}'")
            self.history.clear_all_but_system()
            self.history.add("user", self.plan)
            return self.observation

        return self.waiting_for_plan

    def observation(self):
        self.history.start_transaction()

        # if self.history.get_last_message_role() == "user":
        self.history.add("user", "Ask a single question to better understand the environment.")

        self.history = self.llm.invoke(self.history)

        tool_call = self.history.last()

        logging.info(f"{self.name}: '{tool_call}'")

        self.history.rollback()

        # log the current history
        # logging.info(f"Current history: {self.history}")

        exit()

        return self.execution

    def execution(self):
        if self.history.last_role() != "user":
            self.history.add("user",
                             "Great! Now repeat the next step of the plan verbatim. If you have repeated all the steps in the plan, respond only with 'steps complete'.")

        self.history = self.llm.invoke(self.history)
        step = self.history.last()

        logging.info(f"{self.name}: '{step}'")

        self.task_completed = True

        if "steps complete" in step.lower():
            return self.done_or_error

        return self.observation

    def done_or_error(self):
        self.send_message("ReasoningAgent", Message("user", "done" if self.task_completed else "error"))
        self.task_completed = False

        return self.waiting_for_plan


class ReasoningAgent(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.request_goal)
        self.llm = self.message_bus.get_resource("llm")

    def request_goal(self):
        logging.info(f"{self.name}: Requesting goal")
        return self.planning

    def planning(self):
        system = "You are the Mastermind of a team of agents playing Minecraft. Your goal is to provide the agents with a plan to complete a goal. Your team consists of you and the Motor Control Agent. The Motor Control Agent will execute the plan step by step. Write a short plan for the Motor Control Agent to follow to achieve the goal."
        goal = "Your goal is to: Mine a log."

        history = MessageHistory()
        history.add("system", system)
        history.add("user", goal)

        history = self.llm.invoke(history)

        plan = history.last()
        logging.info(f"{self.name}: Created plan '{plan}'")

        self.send_message("MotorControlAgent", Message("assistant", plan))

        return self.waiting_for_motor

    def waiting_for_motor(self):
        message = self.receive_message(timeout=1)
        if message:
            logging.info(f"{self.name}: Received message '{message.content}'")
            return self.request_goal
        return self.waiting_for_motor


def main():
    # Initialize MessageBus
    bus = MessageBus()

    llm = LLMClient(
        url="http://localhost:1234/v1/chat/completions",
        model="arcee-ai/SuperNova-Medius-GGUF",
        # model="lmstudio-community/qwen2.5-14b-instruct",
        # model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    )

    bus.add_resource("llm", llm)

    # Create agents
    motor_agent = MotorControlAgent("MotorControlAgent", bus)
    reasoning_agent = ReasoningAgent("ReasoningAgent", bus)

    # Start agents
    motor_agent.start()
    reasoning_agent.start()

    # Let the system run for a while
    time.sleep(10)

    # Stop agents
    motor_agent.stop()
    reasoning_agent.stop()

    # Wait for all threads to finish
    motor_agent.join()
    reasoning_agent.join()

    logging.info("System shutdown.")


if __name__ == "__main__":
    main()
