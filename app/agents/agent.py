import os
import uuid
import asyncio
from contextlib import AbstractAsyncContextManager
from typing import AsyncIterable, List, Union, Optional

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    ChatMessage,
    TextContent,
    ChatMessageStore
)

from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import MessageTextContent

class AzureAIAgent(AbstractAsyncContextManager):
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

    async def __aenter__(self):
        if self.project_client:
            await self.project_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.project_client:
            await self.project_client.__aexit__(exc_type, exc_value, traceback)

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

        # Create agent if needed
        model_name = os.environ.get("AZURE_AI_MODEL", "gpt-4o")

        agent = await self.project_client.agents.create_agent(
            model=model_name,
            name="gateway-agent",
            instructions="You are a helpful gateway agent."
        )

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

        if not self.project_client:
             raise ValueError("PROJECT_CONNECTION_STRING not set.")

        agent_id = await self._ensure_agent()
        user_message: str = self._extract_query(messages)

        # Create a thread
        thread_obj = await self.project_client.agents.create_thread()

        await self.project_client.agents.create_message(
            thread_id=thread_obj.id,
            role="user",
            content=user_message
        )

        run = await self.project_client.agents.create_run(
            thread_id=thread_obj.id,
            agent_id=agent_id
        )

        # Poll
        while run.status in ["queued", "in_progress", "requires_action"]:
            await asyncio.sleep(0.5)
            run = await self.project_client.agents.get_run(thread_id=thread_obj.id, run_id=run.id)
            if run.status == "failed":
                response_text = "Error: Agent run failed."
                break
            if run.status == "cancelled":
                response_text = "Error: Agent run cancelled."
                break
            if run.status == "requires_action":
                response_text = "Error: Agent requires action (tool call) which is not supported yet."
                break
        else:
            # Loop finished without break (completed)
            messages_list = await self.project_client.agents.list_messages(thread_id=thread_obj.id)
            # Messages are newest first
            response_text = "No response from agent."
            for msg in messages_list.data:
                if msg.role == "assistant":
                    text_parts = []
                    for content in msg.content:
                        if isinstance(content, MessageTextContent):
                            text_parts.append(content.text.value)
                    response_text = "\n".join(text_parts)
                    break

        response_msg = ChatMessage(
            role="assistant",
            contents=[TextContent(text=response_text)]
        )

        return AgentRunResponse(messages=[response_msg], response_id=str(uuid.uuid4()))

    async def run_stream(
        self,
        messages: Union[List[ChatMessage], str, ChatMessage, List[str]] = None,
        *,
        thread: AgentThread = None,
        **kwargs
    ) -> AsyncIterable[AgentRunResponseUpdate]:

        # Streaming wrapper around buffered run (simplified for this task)
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
