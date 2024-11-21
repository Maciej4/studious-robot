"""
The conversational agent approach has all agents contribute to the same
discussion. This should help improve context and coherence, allowing
agents to better understand and critique each other's arguments.
However, this is done at the cost of the compartmentalization of agents.
"""
import time

from termcolor import colored

from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory
import mine_tools


class Mastermind(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.chat)
        self.llm = self.message_bus.get_resource("llm")
        self.history = self.message_bus.get_resource("history")

        mastermind_system_prompt = (
            "I am the Mastermind agent for a team of agents playing Minecraft. "
            "I write the plans and strategies for the team. "
            "Keep the plans clear, short, and easy to follow. "
            "Once execution of the plan starts, provide guidance to the Actor agent "
            "rather than re-writing the plan. You are not the Critic or the Actor."
            "Do not take on their roles."
        )

        self.history.add("system_" + self.name, mastermind_system_prompt)

    def chat(self):
        self.history = self.llm.invoke(self.history, agent=self.name)
        self.history.set_last_role(self.name)

        last_message = "MASTERMIND: " + self.history.last()
        print(colored(last_message, "green"))

        self.send_message("critic", Message("none", "none"))

        return self.wait

    def wait(self):
        message = self.receive_message(timeout=1)

        if message is None:
            return self.wait

        return self.chat


class Critic(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.wait)
        self.llm = self.message_bus.get_resource("llama")
        self.history = self.message_bus.get_resource("history")

        critic_system_prompt = (
            "I am the Critic agent for a team of agents playing Minecraft. "
            "I critique the plans and strategies written by the Mastermind. "
            "Rather than re-writing the plans, I should focus on finding flaws "
            "and suggesting improvements. The mastermind will then revise the plan. "
            "Once the plan is good, I will approve it by saying 'Plan Complete'."
            "Once execution starts, I will critique the mastermind's guidance "
            "and provide potential explanations and an alternative perspective "
            "to the Actor agent. You are NOT the Mastermind or the Actor. Do not "
            "take on their roles."
        )

        self.history.add("system_" + self.name, critic_system_prompt)

        self.planning_complete = False

    def chat(self):
        self.history = self.llm.invoke(self.history, agent=self.name)
        self.history.set_last_role(self.name)

        last_message = "CRITIC: " + self.history.last()
        print(colored(last_message, "red"))

        if not self.planning_complete:
            if "plan complete" in self.history.last().lower():
                self.planning_complete = True
                return self.plan_complete

            self.send_message("mastermind", Message("none", "none"))
        else:
            self.send_message("actor", Message("none", "none"))

        return self.wait

    def wait(self):
        message = self.receive_message(timeout=1)

        if message is None:
            return self.wait

        return self.chat

    def plan_complete(self):
        print(colored("Plan Complete!", "blue"))

        execution_start_prompt = (
            "Great! Now that the plan is complete, work with the Actor "
            "agent to complete the steps of the plan. The Actor can execute "
            "the plan one tool call at a time. Mastermind and Critic agents, "
            "provide guidance as to what to do and feedback should something "
            "go wrong."
        )

        self.history.add("user", execution_start_prompt)
        print(self.history.last())
        self.send_message("mastermind", Message("none", "none"))
        return self.wait


class Actor(ThreadedAgent):
    def __init__(self, name: str, message_bus: MessageBus):
        super().__init__(name, message_bus, self.wait)
        self.llm = self.message_bus.get_resource("llama")
        self.history = self.message_bus.get_resource("history")

        actor_system_prompt = (
                "I am the Actor agent for a team of agents playing Minecraft. "
                "I execute the plan written by the Mastermind and critiqued by the Critic. "
                "Each of my responses will contain a single tool call, starting at "
                "the beginning of the plan. Do not include any additional information "
                "in your responses. Tool calls are made in the form of python function calls "
                "for instance, visual_question('Describe the scene.'). Tools:\n\n"
                + mine_tools.get_tools_string()
        )

        self.history.add("system_" + self.name, actor_system_prompt)

    def chat(self):
        self.history = self.llm.invoke(self.history, agent=self.name)
        self.history.set_last_role(self.name)

        last_message = self.name.upper() + ": " + self.history.last()
        print(colored(last_message, "yellow"))

        if "task complete" in self.history.last().lower():
            exit()

        tool_call_result = mine_tools.exec_tool_call(self.history.last())

        self.history.add("user", tool_call_result)
        print(self.history.last())

        self.send_message("mastermind", Message("none", "none"))

        return self.wait

    def wait(self):
        message = self.receive_message(timeout=1)

        if message is None:
            return self.wait

        return self.chat


def main():
    bus = MessageBus()

    llm = LLMClient(
        url="http://localhost:1234/v1",
        model="arcee-ai/SuperNova-Medius-GGUF",
        # model="lmstudio-community/qwen2.5-14b-instruct",
    )

    llama = LLMClient(
        url="http://localhost:1234/v1",
        model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    )

    history = MessageHistory()

    overall_system_prompt = (
        "You all are agents in a team playing Minecraft. "
        "Keep you responses as short as possible and avoid repeating what has "
        "already been said."
    )

    history.add("system", overall_system_prompt)

    bus.add_resource("llm", llm)
    bus.add_resource("llama", llama)
    bus.add_resource("history", history)

    Mastermind("mastermind", bus)
    Critic("critic", bus)
    Actor("actor", bus)

    goal = (
        "Write a plan to mine 1 log in Minecraft. "
        "The plan should be clear, short, and easy to follow. "
        "Assume you start with nothing in your inventory."
    )

    history.add("user", goal)

    print(history)

    bus.start_agents()

    while bus.running:
        time.sleep(1)

    bus.stop_agents()


if __name__ == '__main__':
    main()
