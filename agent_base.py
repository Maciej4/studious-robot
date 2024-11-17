import threading
import queue
import json


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
