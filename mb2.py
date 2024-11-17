import time

from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory


class MotorControlAgent(ThreadedAgent):
    def __init__(self, name, message_bus):
        super().__init__(name, message_bus, self.waiting_for_plan)
        self.plan = None
        self.task_completed = False
        self.llm = self.message_bus.get_resource("llm")
        self.history = MessageHistory()
        self.history.add_message("system",
                                 "You are an AI agent playing Minecraft. Repeat the first step of the given plan verbatim.")

    def waiting_for_plan(self):
        message = self.receive_message(timeout=1)

        if message:
            self.plan = message.content
            print(f"{self.name}: Received plan '{self.plan}'")
            self.history.clear_all_but_system()
            self.history.add_message("user", self.plan)
            return self.observation

        return self.waiting_for_plan

    def observation(self):
        time.sleep(1)
        return self.execution

    def execution(self):
        if self.history.get_last_message_role() != "user":
            self.history.add_message("user",
                                     "Great! Now repeat the next step of the plan verbatim. If you have repeated all the steps in the plan, respond only with 'steps complete'.")

        self.history = self.llm.invoke(self.history)
        step = self.history.last()

        print(f"{self.name}: '{step}'")

        self.task_completed = True

        if "steps complete" in step.lower():
            return self.done_or_error

        return self.observation

    def done_or_error(self):
        self.send_message("ReasoningAgent", Message("user", "done" if self.task_completed else "error"))
        self.task_completed = False

        return self.waiting_for_plan


class ReasoningAgent(ThreadedAgent):
    def __init__(self, name, message_bus):
        super().__init__(name, message_bus, self.request_goal)
        self.llm = self.message_bus.get_resource("llm")

    def request_goal(self):
        print(f"{self.name}: Requesting goal")
        return self.planning

    def planning(self):
        system = "You are an AI agent playing Minecraft. Write a simple plan to achieve the given task. Keep your plan short."
        goal = "Mine a log."
        history = MessageHistory()
        history.add_message("system", system)
        history.add_message("user", goal)
        history = self.llm.invoke(history)
        plan = history.last()
        print(f"{self.name}: Created plan '{plan}'")
        self.send_message("MotorControlAgent", Message("assistant", plan))
        return self.waiting_for_motor

    def waiting_for_motor(self):
        message = self.receive_message(timeout=1)
        if message:
            print(f"{self.name}: Received message '{message.content}'")
            return self.request_goal
        return self.waiting_for_motor


# Example usage
if __name__ == "__main__":
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

    print("System shutdown.")
