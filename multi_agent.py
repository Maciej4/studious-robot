import time
import logging
from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory
import mine_tools

logging.basicConfig(level=logging.INFO)


class MotorControlAgent(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.waiting_for_plan)
        self.plan = None
        self.llm = self.message_bus.get_resource("llm")
        self.history = MessageHistory()
        self.tools = mine_tools.get_tools()
        self.tool_string = mine_tools.get_tools_string()

        system = (
            "You are an expert AI agent playing Minecraft. "
            "Given a plan to perform a task, you need to execute the plan step by step."
            # "\nThe tools you can use are listed below.\n"
            # + self.tool_string
        )

        print(system)

        self.history.add("system", system)

    def to_tool_call(self, message: str) -> str:
        """
        Take a message and call an LLM to convert it to a tool call,
        then call the tool and return the result.
        """
        temp_history = MessageHistory()
        temp_system = (
                "You are an expert AI agent playing Minecraft. "
                "Your task is to convert the user's message into a tool call. "
                "Respond with a single tool call python function "
                "that can be used to execute the task described in the user's message. "
                "Do not provide any further information or write any code other "
                "than the single tool function call."
                "The tool call should be in the format tool_name(args). "
                "Do not include (var=something) in the tool call, just "
                "list the arguments in order. "
                "If there are multiple possible tool calls, choose the first one. "
                "Make sure to include parentheses for string arguments."
                "\n\n The tools you can use are listed below:\n\n"
                + self.tool_string
        )
        temp_history.add("system", temp_system)
        temp_history.add("user", message)

        temp_history = self.llm.invoke(temp_history)

        tool_call_message = temp_history.last()

        logging.info(f"{self.name}: '{tool_call_message}'")

        # logging.info(f"{self.name}: '{tool_call_message}'")

        tool_call_result = mine_tools.exec_tool_call(tool_call_message)

        logging.info(f"{self.name}: '{tool_call_result}'")

        return tool_call_result

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

        self.history.add("user",
                         "Ask a single question to better understand the environment. Do not provide any additional information or attempt to execute the plan. Try to ask a different question from those asked in conversation so far. It doesn't have to be completely different, but providing unique perspectives on the environment helps improve performance.")

        self.history = self.llm.invoke(self.history)
        question = self.history.last()
        logging.info(f"{self.name}: '{question}'")

        self.history.rollback()

        self.history.add("assistant", question)
        result = self.to_tool_call(question)
        self.history.add("assistant", result)

        return self.execution

    def execution(self):
        exec_message = (
            "With the above information, describe the next action to take "
            "in order to follow the plan and complete the task. "
            "Respond only with the next action to take, not the whole plan. "
            "This should only be a single step. "
            "Don't speculate on future steps, only focus the next one."
        )

        self.history.add("user", exec_message)

        self.history = self.llm.invoke(self.history)
        step = self.history.last()

        logging.info(f"{self.name}: '{step}'")

        if "steps complete" in step.lower():
            return self.done_or_error

        tool_call_result = self.to_tool_call(step)

        self.history.add("user", tool_call_result)

        return self.observation

    def done_or_error(self):
        self.send_message("ReasoningAgent", Message("user", "done"))

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
    time.sleep(10000)

    # Stop agents
    motor_agent.stop()
    reasoning_agent.stop()

    # Wait for all threads to finish
    motor_agent.join()
    reasoning_agent.join()

    logging.info("System shutdown.")


if __name__ == "__main__":
    main()
