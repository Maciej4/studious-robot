import threading
import queue
import json
import time

from llm_client import LLMClient, MessageHistory


class Message:
    """
    A single message in the message bus.
    """

    def __init__(self, role, content):
        self.role = role  # "user", "assistant", or "system"
        self.content = content  # The message content, a string

    def to_json(self):
        return json.dumps({"role": self.role, "content": self.content})

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return Message(data["role"], data["content"])


class MessageBus:
    """
    A message bus for communication between agents.
    """

    def __init__(self):
        self.agent_queues = {}
        self.agents = {}
        self.lock = threading.Lock()
        self.resources = {}

    def register_agent(self, agent_name, agent):
        with self.lock:
            if agent_name not in self.agent_queues:
                self.agent_queues[agent_name] = queue.Queue()
                self.agents[agent_name] = agent

    def unregister_agent(self, agent_name):
        with self.lock:
            if agent_name in self.agent_queues:
                del self.agent_queues[agent_name]
                del self.agents[agent_name]

    def send_message(self, to_agent, message):
        with self.lock:
            if to_agent in self.agent_queues:
                self.agent_queues[to_agent].put(message)

    def receive_message(self, agent_name, timeout=None):
        with self.lock:
            if agent_name in self.agent_queues:
                agent_queue = self.agent_queues[agent_name]
            else:
                print(f"Agent '{agent_name}' not found.")
                return None
        try:
            message = agent_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def add_resource(self, resource_name, value):
        with self.lock:
            self.resources[resource_name] = value

    def get_resource(self, resource_name):
        with self.lock:
            return self.resources.get(resource_name, None)


class StateMachine:
    def __init__(self, initial_state):
        self.state = initial_state

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def update(self, *args, **kwargs):
        self.state = self.state(*args, **kwargs)


class Agent:
    def __init__(self, name, message_bus):
        self.name = name
        self.message_bus = message_bus
        self.message_bus.register_agent(self.name, self)

    def send_message(self, to_agent, message):
        self.message_bus.send_message(to_agent, message)

    def receive_message(self, timeout=None):
        return self.message_bus.receive_message(self.name, timeout=timeout)

    def stop(self):
        self.message_bus.unregister_agent(self.name)


class ThreadedAgent(Agent, threading.Thread):
    """
    A class for agents that run in loops.
    """

    def __init__(self, name, message_bus, initial_state):
        threading.Thread.__init__(self)  # Initialize the threading.Thread part first
        Agent.__init__(self, name, message_bus)  # Then initialize the Agent part
        self.running = True
        self.sm = StateMachine(initial_state)
        self.past_state = None

    def run(self):
        while self.running:
            state = self.sm.get_state()
            if state != self.past_state:
                self.past_state = state
                print(f"{self.name}: State is '{state.__name__}'")
            self.sm.update()

    def stop(self):
        self.running = False
        super().stop()


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
