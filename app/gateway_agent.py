import uuid
import json
from typing import AsyncIterable, Any, List, Union
from agent_framework import AgentRunResponse, AgentRunResponseUpdate, AgentThread, ChatMessage, ChatMessageStore, TextContent
from app.mock_agents.weather_agent import WeatherAgent
from app.mock_agents.docs_agent import DocsAgent
from app.utils import create_text_response, create_weather_widget, create_doc_results_widget

class GatewayAgent:
    def __init__(self):
        self._id = "gateway-agent"
        self._name = "Gateway Agent"
        self.weather_agent = WeatherAgent()
        self.docs_agent = DocsAgent()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "Gateway Agent"

    @property
    def description(self) -> str:
        return "Orchestrates calls to specialized agents."

    def get_new_thread(self, **kwargs) -> AgentThread:
         return AgentThread(message_store=ChatMessageStore())

    async def run(self, messages: Union[List[ChatMessage], str, ChatMessage, List[str]] = None, *, thread: AgentThread = None, **kwargs: Any) -> AgentRunResponse:
        query = self._extract_query(messages)
        response_payload = await self._process_query(query)

        response_msg = ChatMessage(
            role="assistant",
            contents=[TextContent(text=json.dumps(response_payload))]
        )

        return AgentRunResponse(messages=[response_msg], response_id=str(uuid.uuid4()))

    async def run_stream(self, messages: Union[List[ChatMessage], str, ChatMessage, List[str]] = None, *, thread: AgentThread = None, **kwargs: Any) -> AsyncIterable[AgentRunResponseUpdate]:
        query = self._extract_query(messages)
        response_payload = await self._process_query(query)

        yield AgentRunResponseUpdate(
            text=json.dumps(response_payload),
            response_id=str(uuid.uuid4()),
            role="assistant"
        )

    def _extract_query(self, messages) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, ChatMessage):
            return messages.text
        if isinstance(messages, list):
            if not messages:
                return ""
            last_msg = messages[-1]
            if isinstance(last_msg, str):
                return last_msg
            if hasattr(last_msg, 'text'):
                return last_msg.text
            return str(last_msg)
        return ""

    def _extract_content_str(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # content can be list of TextContent, etc.
            return " ".join([str(c) for c in content])
        return str(content)

    async def _process_query(self, query: str) -> dict:
        query_lower = query.lower()

        if "weather" in query_lower:
            result = self.weather_agent.process(query)
            return create_weather_widget(result["data"])

        elif "doc" in query_lower or "manual" in query_lower or "search" in query_lower or "agent" in query_lower:
            result = self.docs_agent.process(query)
            return create_doc_results_widget(result["data"])

        else:
            return create_text_response(f"I am the Gateway Agent. You said: '{query}'.\nTry asking about 'weather in London' or 'search docs for agent framework'.")
