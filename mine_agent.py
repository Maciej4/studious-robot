from llm_client import LLMClient, MessageHistory
from llm_graph import LLMGraph
from llm_tools import (
    are_tools_present,
    extract_and_run_tools,
    tool,
    tools_to_string,
)


@tool
def look_at(object: str) -> str:
    return input(f"Look at {object}?")


@tool
def move_forward(distance: int) -> str:
    return input(f"Move forward {distance} blocks?")


@tool
def mine_block() -> str:
    return input("Mine the block?")


@tool
def visual_question(question: str) -> str:
    return input(f"Ask the vision model: {question}?")


tools = [look_at, move_forward, mine_block, visual_question]

llm = LLMClient(
    url="http://localhost:1234/v1/chat/completions",
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
)


# Define nodes for the graph
def chatbot(messages: MessageHistory) -> MessageHistory:
    # print(messages)
    response = llm.invoke(messages)
    return response


def route_tools(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    if are_tools_present(last_message):
        return "tools"
    return "END"


def tool_node(messages: MessageHistory):
    last_message = messages.get_last_message_str()
    tool_results = extract_and_run_tools(last_message, tools)
    # tool_result_message = "\n".join(tool_results)
    tool_results = f"```\n{tool_results}\n```\n Tools have been executed. Have you completed your task?"
    messages.add_message("user", tool_results)
    return messages


# Create the graph
graph = LLMGraph()
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(graph.START, "chatbot")
graph.add_edge("tools", "chatbot")
graph.add_conditional_edge("chatbot", route_tools, {"tools": "tools", "END": "END"})

tools_str = tools_to_string(tools)
system_prompt = """You are an AI agent playing Minecraft.\
Provide concise but detailed responses to the user's queries.\
You are working with another language model which has vision capabilities and can see the Minecraft game.\
This model can describe the scene and point out objects, while you need to plan the actions\
to take based on these descriptions in order to achieve a certain goal.\
Describe your thought process, then output a python code block (three backticks, python, code, three backticks)\
with the singular aciton you want to take. Provide only one action at a time to allow for a back-and-forth interaction
with the vision language model.\

HINTS:
- Gather information about the scene using the vision model before deciding on an action.
- Make use of broader questions like visual_question("What is the scene?") to get a general idea of the scene\
before asking more specific questions.
- Use the tools provided to help you decide on an action.

For example, to add 2 and 3, you would write:

```python
add(2, 3)
```

The tools available to you are:
"""

system_prompt += tools_str

print("SYSTEM:", system_prompt)
print()


# Function to stream graph updates
def stream_graph_updates(user_input):
    initial_state = MessageHistory()
    initial_state.add_message("system", system_prompt)
    initial_state.add_message("user", user_input)
    for state in graph.stream(initial_state):
        # print("a")
        print()
        print(state.get_last_message_role().upper() + ":", state.get_last_message_str())
        # print(state)
        # for message in state["messages"]:
        #     print(f"{message['role'].capitalize()}: {message['content']}")


# Main loop
while True:
    user_input = input("USER: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
