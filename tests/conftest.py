import sys
from unittest.mock import MagicMock

# Mock agent_framework if it doesn't exist
try:
    import agent_framework
except ImportError:
    mock_af = MagicMock()
    sys.modules["agent_framework"] = mock_af

    # Define a simple pass-through decorator for ai_function
    def ai_function(func):
        return func

    mock_af.ai_function = ai_function

    # Mock other components used in gateway_agent.py to avoid import errors during collection
    mock_af.AgentRunResponse = MagicMock()
    mock_af.AgentRunResponseUpdate = MagicMock()
    mock_af.AgentThread = MagicMock()
    mock_af.ChatMessage = MagicMock()
    mock_af.ChatMessageStore = MagicMock()
    mock_af.TextContent = MagicMock()
