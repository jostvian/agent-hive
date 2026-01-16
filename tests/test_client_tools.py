from app.tools.client_tools import call_client_tool

def test_call_client_tool():
    tool_name = "display_map"
    args = {"lat": 40.7128, "lon": -74.0060}

    result = call_client_tool(tool_name, args)

    expected_result = {
        "type": "function_call",
        "name": "display_map",
        "arguments": {
            "lat": 40.7128,
            "lon": -74.0060
        }
    }

    assert result == expected_result
