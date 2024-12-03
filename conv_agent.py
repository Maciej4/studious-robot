"""
The conversational agent approach has all agents contribute to the same
discussion. This should help improve context and coherence, allowing
agents to better understand and critique each other's arguments.
However, this is done at the cost of the compartmentalization of agents.
"""
import time

from termcolor import colored
from typing import Literal

from agent_base import ThreadedAgent, MessageBus, Message
from llm_client import LLMClient, MessageHistory
import mine_tools


class ChatAgent(ThreadedAgent):
    """
    An agent which participates in a chat with other agents. It waits
    for a message from another agent, then adds its own message to the
    conversation history and sends a response to the "router" agent.
    """

    # Taken from termcolor._types.py. This is not a good solution, but
    # it does work. This lets the type checker know the possible values
    # for color strings and lets the IDE provide autocompletion.
    Color = Literal[
        "black", "grey", "red", "green", "yellow", "blue", "magenta",
        "cyan", "light_grey", "dark_grey", "light_red", "light_green",
        "light_yellow", "light_blue", "light_magenta", "light_cyan",
        "white",
    ]

    def __init__(self, name: str,
                 message_bus: MessageBus,
                 text_color: Color | None,
                 llm_name: str,
                 system_prompt: str):
        super().__init__(name, message_bus, self.chat)
        self.llm: LLMClient = self.message_bus.get_resource(llm_name)
        self.history: MessageHistory = self.message_bus.get_resource("history")

        self.history.add("system_" + self.name, system_prompt)

        self.text_color = text_color

    def chat(self):
        last_message = self.receive_message(timeout=1)

        if last_message is None:
            return self.chat

        self.history = self.llm.invoke(self.history, agent=self.name)
        self.history.set_last_role(self.name)

        last_message = self.name.upper() + ": " + self.history.last()
        print(colored(last_message, self.text_color))

        self.send_message("router", Message("none", "none"))

        return self.chat


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


class Router(ThreadedAgent):
    # The router agent will be responsible for deciding which agent
    # should be the next to speak based on the current conversation state.
    def __init__(self, message_bus: MessageBus):
        super().__init__("router", message_bus, self.simple_route)
        self.llm: LLMClient = self.message_bus.get_resource("llama")
        self.history: MessageHistory = self.message_bus.get_resource("history")

        system_prompt = (
            "You are the Router agent. You decide which agent speaks next."
            "The agents in the conversation are the Mastermind, Critic, and Observer."
            "Analyse the conversation and decide which agent would be the best to speak "
            "next based on the current conversation state. You must respond only with "
            "a single agent's name: 'mastermind', 'critic', or 'observer'. "
            "Do not provide any additional information or context. Do not call "
            "the same agent twice in a row.\n\n"
            "- Mastermind: Writes the plans and strategies for the team.\n"
            "- Critic: Critiques the plans and strategies written by the Mastermind.\n"
            "- Observer: Can see the scene and perform visual reasoning.\n\n"
            "Do not write any plans or strategies yourself, just decide which agent speaks next."
        )

        self.history.add("system_router", system_prompt)

        self.agent_index = 0
        # self.agents = ["mastermind", "observer", "critic", "mover"]
        self.agents = ["mastermind", "critic", "mover"]

    def simple_route(self):
        # Route the agents in a round-robin fashion
        agent = self.agents[self.agent_index]
        self.agent_index = (self.agent_index + 1) % len(self.agents)
        self.send_message(agent, Message("none", "none"))
        return self.wait

    def wait(self):
        message = self.receive_message(timeout=1)

        if message is None:
            return self.wait

        return self.simple_route


class Mastermind2(ChatAgent):
    # The mastermind agent will write the plans and strategies for the team.
    def __init__(self, message_bus: MessageBus):
        system_prompt = (
            "You are the Mastermind agent. You write the plans and "
            "strategies for the team. Once the plan has been finalized, "
            "focus on going step-by-step through the plan. "
            "Ask the Observer agent for information about the scene."
        )

        super().__init__("mastermind", message_bus, "green", "llm", system_prompt)


class Critic2(ChatAgent):
    # The critic agent will critique the plans and strategies written by the mastermind.
    def __init__(self, message_bus: MessageBus):
        system_prompt = (
            "You are the Critic agent. You critique the plans "
            "and strategies written by the Mastermind and consider "
            "alternative perspectives. Once the plan is good, approve it "
            "and move on to working with the other agents in executing the plan."
        )

        super().__init__("critic", message_bus, "red", "llama", system_prompt)


# class Observer(ChatAgent):
#     # The viewer agent will connect to Molmo through the control API
#     # and thus will be able to see the scene and perform visual reasoning.
#     def __init__(self, message_bus: MessageBus):
#         system_prompt = (
#             "You are the Observer agent. You can see the scene and describe what is happening "
#             "in the Minecraft environment (not in the conversation). "
#             "Do not attempt to write plans or perform critiques yourself, just provide visual "
#             "information. You can see a forest 20 blocks in front of you. There are no mobs in sight. "
#             "You are not the Mastermind nor the Critic. Do not take on their roles."
#             "If you don't have anything relevant to say, simply say 'nothing to report'."
#         )
#
#         super().__init__("observer", message_bus, "yellow", "llama", system_prompt)


class Observer2(ThreadedAgent):
    # This observer will call the visual_question tool to describe the scene.
    def __init__(self, message_bus: MessageBus):
        super().__init__("observer", message_bus, self.wait)
        self.llm: LLMClient = self.message_bus.get_resource("llm")
        self.history: MessageHistory = self.message_bus.get_resource("history")

        system_prompt = (
            "You are the Observer agent. Describe the Minecraft scene in front of you. "
            "Answer any questions the other agents have, but don't take on their roles. "
            "Don't write plans or critiques, just provide visual information. "
            "Focus on details relevant to the conversation and ignore those that are not."
        )

        self.history.add("system_" + self.name, system_prompt)

    def wait(self):
        last_message = self.receive_message(timeout=1)

        if last_message is None:
            return self.wait

        self.history.start_transaction()

        ask_molmo_prompt = (
            "Look at the earlier conversation to find any unanswered questions "
            "or details to gather about the Minecraft scene. Then write a short "
            "question or two that would be relevant to the conversation. "
            "Don't write any plans or critiques, just focus on finding gaps "
            "in the knowledge of the game environment. "
            "Distance to objects is a particularly useful piece of information, "
            "as well as the presence of any relevant objects."
        )

        self.history.add("user", ask_molmo_prompt)
        self.history = self.llm.invoke(self.history, agent=self.name)

        question = self.history.last()
        print(colored("OBSERVER: " + question, "light_yellow"))

        self.history.rollback()

        tool_result = mine_tools.visual_question(question)

        self.history.add(self.name, tool_result)

        last_message = self.name.upper() + ": " + self.history.last()
        print(colored(last_message, "yellow"))

        self.send_message("router", Message("none", "none"))
        return self.wait


class Watchdog(ThreadedAgent):
    # The watchdog agent will monitor the conversation and ensure that
    # the agents are following the conversation rules and staying on topic.
    # In addition, the watchdog will terminate the conversation once the
    # task is complete.
    def __init__(self, message_bus: MessageBus):
        super().__init__("watchdog", message_bus, self.wait)
        self.history: MessageHistory = self.message_bus.get_resource("history")
        self.llm: LLMClient = self.message_bus.get_resource("llama")

        system_prompt = (
            "You are the Watchdog agent. You monitor the conversation and ensure that "
            "the agents are following the conversation rules and staying on topic. "
            "If the conversation goes off-topic or the agents are not following the rules, "
            "you can intervene and correct them. Once the task "
            "is complete, you will end the conversation."
            "Respond only with a score of 1-10, where 1 is the worst "
            "(off-topic, distracted, stuck) and 10 is the best "
            "(working towards the goal). "
            "Respond with an '11' to end the conversation."
        )

        self.history.add("system_" + self.name, system_prompt)

    def look(self):
        self.history.start_transaction()

        look_prompt = (
            "How is the conversation going? Rate it from 1-10 or "
            "end it with an '11' if the task has been achieved. "
            "Do not provide any other feedback, context, or information."
        )

        self.history.add("user", look_prompt)
        self.history = self.llm.invoke(self.history, agent=self.name)
        last_message = self.history.last()

        if "11" in last_message:
            print("Conversation ended by Watchdog.")
            exit()

        print(colored("WATCHDOG: " + last_message, "cyan"))

        self.history.rollback()
        self.send_message("router", Message("none", "none"))
        return self.wait

    def wait(self):
        last_message = self.receive_message(timeout=1)

        if last_message is None:
            return self.wait

        return self.look


class Mover(ThreadedAgent):
    # The mover agent will execute the plan written by the mastermind.
    def __init__(self, message_bus: MessageBus):
        super().__init__("mover", message_bus, self.wait)
        self.llm: LLMClient = self.message_bus.get_resource("llm")
        self.history: MessageHistory = self.message_bus.get_resource("history")

        system_prompt = (
            "You are the Mover agent. You execute the plan written by the Mastermind. "
            "Each of your responses will contain a single tool call, starting at "
            "the beginning of the plan. Do not include any additional information "
            "in your responses."
        )

        self.history.add("system_" + self.name, system_prompt)

    def chat(self):
        self.history.start_transaction()

        mover_prompt = (
            "Execute the plan written by the Mastermind agent. "
            "Respond with only a single tool call at a time. "
        )

        self.history.add("user", mover_prompt)

        self.history = self.llm.invoke(self.history, agent=self.name, max_tokens=100)
        self.history.set_last_role(self.name)

        last_message = self.name.upper() + ": " + self.history.last()
        print(colored(last_message, "magenta"))

        if "waiting" not in self.history.last().lower():
            tool_call_result = mine_tools.exec_tool_call(self.history.last())

            self.history.rollback()

            self.history.add("user", tool_call_result)
            print(self.history.last())
        else:
            self.history.rollback()

        self.send_message("router", Message("none", "none"))

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
            "Keep you responses as short and work together to complete the task. "
            "If you want some information about the scene, ask the Observer agent. "
            "If you want to get a plan, ask the Mastermind agent. "
            "If you want to get a critique ask the Critic agent. "
            "Do not assume the roles of other agents. "
            "Tool calls are made in the form of python function calls "
            "for instance, visual_question('Describe the scene.'). "
            "When making a tool call, provide the function name and arguments. "
            "Including argname=value pairs is not permitted and will cause "
            "the tool call to fail. "
            "The tools you can use are: "
            + mine_tools.get_tools_string()
    )

    history.add("system", overall_system_prompt)

    bus.add_resource("llm", llm)
    bus.add_resource("llama", llama)
    bus.add_resource("history", history)

    # Mastermind("mastermind", bus)
    # Critic("critic", bus)
    # Actor("actor", bus)

    Router(bus)
    Mastermind2(bus)
    Critic2(bus)
    Observer2(bus)
    Mover(bus)
    Watchdog(bus)

    goal = (
        "Write a plan to mine 1 log in Minecraft. "
        "The plan should be clear, short, and easy to follow. "
        "Assume you start with nothing in your inventory. "
        "The actions you can take are: look at an object in sight, "
        "move forward, or mine a block in front of you."
    )

    history.add("user", goal)

    print(history)

    bus.start_agents()

    while bus.running:
        time.sleep(1)

    bus.stop_agents()


if __name__ == '__main__':
    main()
