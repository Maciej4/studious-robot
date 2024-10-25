class LLMGraph:
    START = "START"
    END = "END"

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

    def add_conditional_edge(self, from_node, condition_func, condition_map):
        self.conditional_edges[from_node] = (condition_func, condition_map)

    def stream(self, initial_state):
        current_node = "START"
        state = initial_state

        while current_node != "END":
            if current_node in self.conditional_edges:
                condition_func, condition_map = self.conditional_edges[current_node]
                next_node_key = condition_func(state)
                current_node = condition_map.get(next_node_key, "END")
            else:
                next_nodes = self.edges.get(current_node, ["END"])
                current_node = next_nodes[0] if next_nodes else "END"

            if current_node in self.nodes:
                state = self.nodes[current_node](state)
                yield state

        # yield state


def main():
    # Example usage
    def chatbot(state):
        user_message = state["messages"][-1]
        response = {
            "messages": [
                {"role": "assistant", "content": f"Echo: {user_message['content']}"}
            ]
        }
        return response

    def route_tools(state):
        last_message = state["messages"][-1]
        if "tool_calls" in last_message:
            return "tools"
        return "END"

    def tool_node(state):
        last_message = state["messages"][-1]
        tool_result = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Tool result for: {last_message['content']}",
                }
            ]
        }
        return tool_result

    graph = LLMGraph()
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)
    graph.add_edge(graph.START, "chatbot")
    graph.add_edge("tools", "chatbot")
    graph.add_conditional_edge("chatbot", route_tools, {"tools": "tools", "END": "END"})

    def stream_graph_updates(user_input):
        initial_state = {"messages": [{"role": "user", "content": user_input}]}
        for state in graph.stream(initial_state):
            for message in state["messages"]:
                print(f"{message['role'].capitalize()}: {message['content']}")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)


if __name__ == "__main__":
    main()
