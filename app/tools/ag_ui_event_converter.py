from typing import List, Optional, Dict, Set, Any
from ag_ui.core.events import (
    BaseEvent,
    EventType,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from openai.types.chat import ChatCompletionChunk

class AGUIEventConverter:
    """
    A generic converter to transform OpenAI streaming events (ChatCompletionChunk)
    into AG_UI events.
    """
    def __init__(self):
        self.started_message_ids: Set[str] = set()
        self.active_tool_calls: Dict[int, str] = {} # index -> tool_call_id

    def convert_event(self, chunk: ChatCompletionChunk) -> List[BaseEvent]:
        """
        Converts an OpenAI ChatCompletionChunk into a list of AG_UI BaseEvents.
        """
        events: List[BaseEvent] = []

        if not chunk.choices:
            return events

        choice = chunk.choices[0]
        delta = choice.delta
        message_id = chunk.id

        # Dispatch to handlers
        self._process_text_content(delta, message_id, events)
        self._process_tool_calls(delta, message_id, events)
        self._process_finish_reason(choice, message_id, events)

        return events

    def _process_text_content(self, delta: Any, message_id: str, events: List[BaseEvent]):
        """Handles text content updates."""
        if delta.content is not None and delta.content != "":
            # Start message if not already started
            if message_id not in self.started_message_ids:
                events.append(TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role="assistant"
                ))
                self.started_message_ids.add(message_id)

            # Add content
            events.append(TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=delta.content
            ))

    def _process_tool_calls(self, delta: Any, message_id: str, events: List[BaseEvent]):
        """Handles tool call updates."""
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                index = tool_call.index

                # If we have an ID, it's the start of a new tool call
                if tool_call.id:
                    # In OpenAI stream, only the first chunk of a tool call has the ID and name
                    tool_call_id = tool_call.id
                    name = tool_call.function.name if tool_call.function and tool_call.function.name else "unknown_tool"

                    self.active_tool_calls[index] = tool_call_id

                    events.append(ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tool_call_id,
                        tool_call_name=name,
                        parent_message_id=message_id # Linking tool call to the completion message ID
                    ))

                # If we have arguments, it's a content update for the tool call
                if tool_call.function and tool_call.function.arguments:
                    tool_call_id = self.active_tool_calls.get(index)
                    if tool_call_id:
                        events.append(ToolCallArgsEvent(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=tool_call_id,
                            delta=tool_call.function.arguments
                        ))

    def _process_finish_reason(self, choice: Any, message_id: str, events: List[BaseEvent]):
        """Handles finish reasons."""
        if choice.finish_reason:
            if choice.finish_reason == "stop":
                # If we were streaming text, end it
                if message_id in self.started_message_ids:
                    events.append(TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id
                    ))
                    # Cleanup
                    self.started_message_ids.discard(message_id)

            elif choice.finish_reason == "tool_calls":
                # End all active tool calls
                for index, tool_call_id in list(self.active_tool_calls.items()):
                    events.append(ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_call_id
                    ))
                self.active_tool_calls.clear()

            # Note: There might be other finish reasons like "length" or "content_filter"
            # which we might want to map to an error or just end the message.
            # For now, we treat them as ending the message if it was started.
            elif message_id in self.started_message_ids:
                 events.append(TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id
                    ))
                 self.started_message_ids.discard(message_id)
