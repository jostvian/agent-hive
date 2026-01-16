from agent_framework import ai_function

@ai_function
def call_client_tool(tool_name: str, args: dict) -> dict:
    """
    Use this tool to trigger a client-side function call.

    Args:
        tool_name (str): The name of the tool to call on the client side.
        args (dict): The arguments to pass to the client-side tool.

    Returns:
        dict: An ag_ui event dictionary of type 'function_call'.
    """
    return {
        "type": "function_call",
        "name": tool_name,
        "arguments": args
    }
