import re
import ast
import inspect
from typing import Callable, List, Tuple, Any


# Custom class to wrap tool functions
class Tool:
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.signature = inspect.signature(func)
        self.docstring = inspect.getdoc(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return f"{self.name}{self.signature} - {self.docstring}"

    def __repr__(self):
        return f"{self.name}{self.signature}"


# Custom decorator to mark functions as tools
def tool(func: Callable) -> Tool:
    return Tool(func)


# Function to print the name, signature, and docstring of each tool
def print_tools_info(tools: List[Tool]):
    for tool in tools:
        print(tool)


# Tools to string
def tools_to_string(tools: List[Tool]) -> str:
    tool_strings = [str(tool) for tool in tools]
    return "\n".join(tool_strings)


# Convert a list of tools to a dictionary with the tool name as the key
def tools_to_dict(tools: List[Tool]) -> dict:
    return {tool.name: tool for tool in tools}


# Function to parse Python code
def parse_python_code(
    code: str, valid_functions: List[str]
) -> List[Tuple[str, Tuple[Any, ...]]]:
    tree = ast.parse(code)
    parsed_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in valid_functions:
                args = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant):  # For Python 3.8+
                        args.append(arg.value)
                    elif isinstance(arg, ast.Str):  # For Python 3.7 and below
                        args.append(arg.s)
                    elif isinstance(arg, ast.Num):  # For Python 3.7 and below
                        args.append(arg.n)
                parsed_calls.append((func_name, tuple(args)))

    return parsed_calls


# Function to verify function call matches the signature
def verify_function_call(tool: Tool, args: Tuple[Any, ...]) -> bool:
    try:
        tool.signature.bind(*args)
        return True
    except TypeError:
        return False


# Function to execute the function call
def execute_function_call(tool: Tool, args: Tuple[Any, ...]) -> Any:
    return tool(*args)


def execute_tools(
    parsed_code: List[Tuple[str, Tuple[Any, ...]]], tools_dict: dict
) -> str:
    outputs = []

    for func_name, args in parsed_code:
        if func_name in tools_dict:
            tool = tools_dict[func_name]
            if verify_function_call(tool, args):
                result = str(execute_function_call(tool, args))
                outputs.append(f"{func_name}{args} -> {result}")
            else:
                outputs.append(f"Invalid arguments for function {func_name}: {args}")
                # raise TypeError(f"Invalid arguments for function {func_name}: {args}")
        else:
            outputs.append(f"Function {func_name} is not a valid tool")
            # raise NameError(f"Function {func_name} is not a valid tool")

        break

    return "\n".join(outputs)


def are_tools_present(message: str) -> bool:
    # Check if the message contains a python code block
    # which must be enclosed in triple quotes using regex
    # return bool(re.search(r"```(.*?)```", message, re.DOTALL))

    # Instead check for a function call of the form function(arg1, arg2, ...)

    # return bool(re.search(r"\w+\(.*\)", message))

    return bool(re.search(r"\w+?\(.*?\)", message))


def extract_and_run_tools(message: str, tools: List[Tool]) -> str:
    # Extract the python code block from the message
    # code_blocks = re.findall(r"```(.*?)```", message, re.DOTALL)

    # Find the first function call in the message
    # code_blocks = re.findall(r"\w+\(.*\)", message)

    code_blocks = re.findall(r"\w+?\(.*?\)", message, re.DOTALL)

    # List of valid tool function names
    valid_functions = [tool.name for tool in tools]

    tool_results = []

    if not code_blocks:
        return "No tools found in the message."

    # for code in code_blocks:
    code = code_blocks[0]
    parsed_code = parse_python_code(code, valid_functions)
    tools_dict = tools_to_dict(tools)
    result = execute_tools(parsed_code, tools_dict)
    tool_results.append(result)

    return "\n".join(tool_results)


def main():
    # Example tools
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    @tool
    def add(x: float, y: float) -> float:
        """Add two numbers."""
        return x + y

    @tool
    def visual_question(question: str) -> str:
        """Ask a question to the visual language model."""
        return "Visual question answer"

    @tool
    def mine_block() -> None:
        return

    @tool
    def move_forward(distance: int) -> None:
        return

    # List of tools
    tools = [multiply, add, visual_question, mine_block, move_forward]

    # Example usage
    code = """
    # Collect the floating oak log
    visual_question("Is there anything blocking us from collecting the floating oak log item?")

    # If not, collect it!
    mine_block() # Execute a special action to pick up the floating item

    # Explore the surrounding trees for more logs or resources
    move_forward(5) # Move forward a bit to get a better view of the surrounding area

    move_forward("abc") # Move forward a bit more to explore further

    visual_question("Are there any other visible logs, chests, or structures within our current view range?")
    """

    valid_functions = ["visual_question", "mine_block", "move_forward"]

    parsed_code = parse_python_code(code, valid_functions)

    # Example tools for testing
    tools_dict = {tool.name: tool for tool in tools}

    # Verify and execute parsed function calls
    execute_tools(parsed_code, tools_dict)


if __name__ == "__main__":
    main()
