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
        self.transitions = {}

    def add_transition(self, state_from, state_to, condition=lambda: True):
        if state_from not in self.transitions:
            self.transitions[state_from] = []
        self.transitions[state_from].append((condition, state_to))

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def update(self, *args, **kwargs):
        if self.state in self.transitions:
            for condition, state_to in self.transitions[self.state]:
                if condition(*args, **kwargs):
                    self.state = state_to
                    break


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

    def __init__(self, name, message_bus):
        threading.Thread.__init__(self)  # Initialize the threading.Thread part first
        Agent.__init__(self, name, message_bus)  # Then initialize the Agent part
        self.running = True

    def run(self):
        # To be implemented by subclasses
        pass

    def stop(self):
        self.running = False
        super().stop()


class MotorControlAgent(ThreadedAgent):
    def __init__(self, name, message_bus):
        super().__init__(name, message_bus)
        self.plan = None
        self.sm = StateMachine("waiting_for_plan")
        self.setup_state_machine()
        self.task_completed = False
        self.past_state = None
        self.llm = self.message_bus.get_resource("llm")
        self.history = MessageHistory()
        self.history.add_message("system",
                                 "You are an AI agent playing Minecraft. Repeat the first step of the given plan verbatim.")

    def setup_state_machine(self):
        self.sm.add_transition("waiting_for_plan", "observation", self.plan_received)
        self.sm.add_transition("observation", "done_or_error", self.plan_complete_or_error)
        self.sm.add_transition("done_or_error", "waiting_for_plan")
        self.sm.add_transition("observation", "execution", self.observation_done)
        self.sm.add_transition("execution", "observation", self.execution_done)

    def plan_received(self, *args, **kwargs):
        return self.plan is not None

    def observation_done(self, *args, **kwargs):
        return True  # Simulate observation done

    def execution_done(self, *args, **kwargs):
        return True  # Simulate execution done

    def plan_complete_or_error(self, *args, **kwargs):
        return "steps complete" in self.history.last().lower()

    def run(self):
        while self.running:
            state = self.sm.get_state()

            if state != self.past_state:
                self.past_state = state
                print(f"{self.name}: State is '{state}'")

            if state == "waiting_for_plan":
                self.wait_for_plan()
            elif state == "observation":
                self.observe()
            elif state == "execution":
                self.execute()
            elif state == "done_or_error":
                self.send_message("ReasoningAgent", Message("user", "done" if self.task_completed else "error"))
                self.task_completed = False
                self.sm.update()

    def wait_for_plan(self):
        message = self.receive_message(timeout=1)
        if message:
            self.plan = message.content
            print(f"{self.name}: Received plan '{self.plan}'")
            self.sm.update()
            self.history.clear_all_but_system()
            self.history.add_message("user", self.plan)

    def observe(self):
        # Simulate user input observation
        print(f"{self.name}: Observing environment")
        time.sleep(1)

        self.sm.update()

    def execute(self):
        # Simulate plan execution
        print(f"{self.name}: Executing plan step")

        if self.history.get_last_message_role() != "user":
            self.history.add_message("user",
                                     "Great! Now repeat the next step of the plan verbatim. If you have repeated all the steps in the plan, respond only with 'steps complete'.")

        self.history = self.llm.invoke(self.history)
        step = self.history.last()

        print(f"{self.name}: Plan step '{step}' executed")

        self.task_completed = True
        self.sm.update()


class ReasoningAgent(ThreadedAgent):
    def __init__(self, name, message_bus):
        super().__init__(name, message_bus)
        self.llm = self.message_bus.get_resource("llm")
        self.sm = StateMachine("request_goal")
        self.setup_state_machine()
        self.past_state = None

    def setup_state_machine(self):
        self.sm.add_transition("request_goal", "planning", self.goal_received)
        self.sm.add_transition("planning", "waiting_for_motor", self.plan_created)
        self.sm.add_transition("waiting_for_motor", "request_goal", self.replan_requested)

    def goal_received(self, *args, **kwargs):
        return True  # Simulate goal received from command line input

    def plan_created(self, *args, **kwargs):
        return True  # Simulate plan creation

    def replan_requested(self, *args, **kwargs):
        return True  # Simulate replan request from MotorControlAgent

    def run(self):
        while self.running:
            state = self.sm.get_state()

            if state != self.past_state:
                self.past_state = state
                print(f"{self.name}: State is '{state}'")

            if state == "request_goal":
                self.request_goal()
            elif state == "planning":
                self.create_plan()
            elif state == "waiting_for_motor":
                self.wait_for_motor()

    def request_goal(self):
        # Simulate reading goal from command line
        print(f"{self.name}: Requesting goal")
        self.sm.update()

    def create_plan(self):
        # Generate a plan
        system = "You are an AI agent playing Minecraft. Write a simple plan to achieve the given task. Keep your plan short."
        goal = "Mine a log."

        # Create a message history
        history = MessageHistory()
        history.add_message("system", system)
        history.add_message("user", goal)

        # Invoke the LLM model
        history = self.llm.invoke(history)

        # Extract the plan from the last message
        plan = history.last()

        print(f"{self.name}: Created plan '{plan}'")
        self.send_message("MotorControlAgent", Message("assistant", plan))
        self.sm.update()

    def wait_for_motor(self):
        message = self.receive_message(timeout=1)
        if message:
            print(f"{self.name}: Received message '{message.content}'")
            self.sm.update()


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
    time.sleep(20)

    # Stop agents
    motor_agent.stop()
    reasoning_agent.stop()

    # Wait for all threads to finish
    motor_agent.join()
    reasoning_agent.join()

    print("System shutdown.")
