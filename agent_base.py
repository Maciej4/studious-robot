import threading
import queue
import json
import logging
from typing import Optional, Any

logging.basicConfig(level=logging.INFO)


class Message:
    """
    A single message in the message bus.
    """

    def __init__(self, role: str, content: str):
        self.role = role  # "user", "assistant", or "system"
        self.content = content  # The message content, a string

    def to_json(self) -> str:
        return json.dumps({"role": self.role, "content": self.content})

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @staticmethod
    def from_json(json_str: str) -> 'Message':
        data = json.loads(json_str)
        return Message(data["role"], data["content"])

    def copy(self):
        return self.__copy__()

    def __str__(self):
        return f"{self.role.upper()}: {self.content}"

    def __repr__(self):
        return f"Message(role='{self.role}', content='{self.content}')"

    def __copy__(self):
        return Message(self.role, self.content)


class MessageBus:
    """
    A message bus for communication between agents.
    """

    def __init__(self):
        self.agent_queues = {}
        self.agents = {}
        self.lock = threading.Lock()
        self.resources = {}

    def register_agent(self, agent_name: str, agent: 'Agent'):
        with self.lock:
            if agent_name not in self.agent_queues:
                self.agent_queues[agent_name] = queue.Queue()
                self.agents[agent_name] = agent

    def unregister_agent(self, agent_name: str):
        with self.lock:
            if agent_name in self.agent_queues:
                del self.agent_queues[agent_name]
                del self.agents[agent_name]

    def send_message(self, to_agent: str, message: Message):
        with self.lock:
            if to_agent in self.agent_queues:
                self.agent_queues[to_agent].put(message)

    def receive_message(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        with self.lock:
            if agent_name in self.agent_queues:
                agent_queue = self.agent_queues[agent_name]
            else:
                logging.warning(f"Agent '{agent_name}' not found.")
                return None
        try:
            message = agent_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def add_resource(self, resource_name: str, value: Any):
        with self.lock:
            self.resources[resource_name] = value

    def get_resource(self, resource_name: str) -> Any:
        with self.lock:
            return self.resources.get(resource_name, None)


class StateMachine:
    def __init__(self, initial_state: callable):
        self.state = initial_state

    def set_state(self, state: callable):
        self.state = state

    def get_state(self) -> callable:
        return self.state

    def update(self, *args, **kwargs):
        self.state = self.state(*args, **kwargs)


class Agent:
    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.message_bus = message_bus
        self.message_bus.register_agent(self.name, self)

    def send_message(self, to_agent: str, message: Message):
        self.message_bus.send_message(to_agent, message)

    def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        return self.message_bus.receive_message(self.name, timeout=timeout)

    def stop(self):
        self.message_bus.unregister_agent(self.name)


class ThreadedAgent(Agent, threading.Thread):
    """
    A class for agents that run in loops.
    """

    def __init__(self, name: str, message_bus: MessageBus, initial_state: callable):
        threading.Thread.__init__(self)  # Initialize the threading. Must be done first.
        Agent.__init__(self, name, message_bus)  # Then initialize the Agent part.
        self.running = True
        self.sm = StateMachine(initial_state)
        self.past_state = None

    def run(self):
        try:
            while self.running:
                state = self.sm.get_state()
                if state != self.past_state:
                    self.past_state = state
                    logging.info(f"{self.name}: Entered state '{state.__name__}'")
                self.sm.update()
        except Exception as e:
            logging.error(f"{self.name}: {e}")

    def stop(self):
        self.running = False
        super().stop()
