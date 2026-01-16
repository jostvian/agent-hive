import os
import uuid
import asyncio
import time
from typing import AsyncIterable, List, Union, Optional

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    ChatMessage,
    TextContent,
    ChatMessageStore
)

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import MessageTextContent

class AzureAIAgent:
    def __init__(self) -> None:
        self._id: str = "azure-ai-gateway-agent"
        self._name: str = "Azure AI Gateway Agent"

        # Take the project endpoint from environment
        connection_string: Optional[str] = os.environ.get("PROJECT_CONNECTION_STRING")
        if connection_string:
            self.project_client: Optional[AIProjectClient] = AIProjectClient.from_connection_string(
                conn_str=connection_string,
                credential=DefaultAzureCredential()
            )
        else:
            self.project_client = None

        self.agent_id: Optional[str] = os.environ.get("AZURE_AI_AGENT_ID")
        self._agent_obj = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Agent backed by Azure AI Foundry"

    async def _ensure_agent(self) -> str:
        if self.agent_id:
            return self.agent_id

        if self._agent_obj:
             return self._agent_obj.id

        if not self.project_client:
            raise ValueError("PROJECT_CONNECTION_STRING not set.")

        def create_agent_sync():
            # NOTE: Do not use 'with self.project_client' as it closes the client.
            # We need a model. "gpt-4o" is a safe bet for Azure AI Foundry defaults
            return self.project_client.agents.create_agent(
                model="gpt-4o",
                name="gateway-agent",
                instructions="You are a helpful gateway agent."
            )

        agent = await asyncio.to_thread(create_agent_sync)
        self._agent_obj = agent
        self.agent_id = agent.id
        return agent.id

    def get_new_thread(self, **kwargs) -> AgentThread:
        return AgentThread(message_store=ChatMessageStore())

    async def run(
        self,
        messages: Union[List[ChatMessage], str, ChatMessage, List[str]] = None,
        *,
        thread: AgentThread = None,
        **kwargs
    ) -> AgentRunResponse:

        agent_id = await self._ensure_agent()
        user_message: str = self._extract_query(messages)

        response_text = await asyncio.to_thread(self._run_sync, agent_id, user_message)

        response_msg = ChatMessage(
            role="assistant",
            contents=[TextContent(text=response_text)]
        )

        return AgentRunResponse(messages=[response_msg], response_id=str(uuid.uuid4()))

    def _run_sync(self, agent_id: str, user_message: str) -> str:
        if not self.project_client:
             raise ValueError("PROJECT_CONNECTION_STRING not set.")

        # Create a thread
        thread_obj = self.project_client.agents.create_thread()

        self.project_client.agents.create_message(
            thread_id=thread_obj.id,
            role="user",
            content=user_message
        )

        run = self.project_client.agents.create_run(
            thread_id=thread_obj.id,
            agent_id=agent_id
        )

        # Poll
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(0.5)
            run = self.project_client.agents.get_run(thread_id=thread_obj.id, run_id=run.id)
            if run.status == "failed":
                return "Error: Agent run failed."
            if run.status == "cancelled":
                return "Error: Agent run cancelled."
            if run.status == "requires_action":
                # Break to avoid infinite loop
                return "Error: Agent requires action (tool call) which is not supported yet."

        messages = self.project_client.agents.list_messages(thread_id=thread_obj.id)
        # Messages are newest first
        for msg in messages.data:
            if msg.role == "assistant":
                text_parts = []
                for content in msg.content:
                    if isinstance(content, MessageTextContent):
                        text_parts.append(content.text.value)
                return "\n".join(text_parts)
        return "No response from agent."

    async def run_stream(
        self,
        messages: Union[List[ChatMessage], str, ChatMessage, List[str]] = None,
        *,
        thread: AgentThread = None,
        **kwargs
    ) -> AsyncIterable[AgentRunResponseUpdate]:

        # Streaming wrapper around buffered run
        response = await self.run(messages, thread=thread, **kwargs)
        if hasattr(response, 'messages') and response.messages:
             for msg in response.messages:
                 if msg.contents:
                     text_content = ""
                     for content in msg.contents:
                         if isinstance(content, TextContent):
                             text_content += content.text
                         elif hasattr(content, 'text'):
                             text_content += content.text
                         else:
                             text_content += str(content)

                     yield AgentRunResponseUpdate(
                         text=text_content,
                         response_id=response.response_id,
                         role="assistant"
                     )

    def _extract_query(self, messages: Union[List[ChatMessage], str, ChatMessage, List[str]]) -> str:
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
