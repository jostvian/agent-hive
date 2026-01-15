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
from openai.types.responses import ResponseStreamEvent

class AGUIEventConverter:
    """
    A generic converter to transform OpenAI Responses API streaming events (ResponseStreamEvent)
    into AG_UI events.
    """
    def __init__(self):
        # Maps item_id (from OpenAI) to tool_call_id (for AG_UI/OpenAI correlation)
        self.item_id_to_tool_call_id: Dict[str, str] = {}

    def convert_event(self, event: ResponseStreamEvent) -> List[BaseEvent]:
        """
        Converts an OpenAI ResponseStreamEvent into a list of AG_UI BaseEvents.
        """
        events: List[BaseEvent] = []

        if event.type == "response.output_item.added":
            self._process_output_item_added(event, events)

        elif event.type == "response.output_text.delta":
            self._process_text_delta(event, events)

        elif event.type == "response.output_text.done":
            self._process_text_done(event, events)

        elif event.type == "response.function_call_arguments.delta":
            self._process_function_call_delta(event, events)

        elif event.type == "response.function_call_arguments.done":
            self._process_function_call_done(event, events)

        return events

    def _process_output_item_added(self, event: Any, events: List[BaseEvent]):
        """Handles new output items (messages or tool calls)."""
        item = event.item

        if item.type == "message":
            if item.role == "assistant":
                # Start of a text message
                events.append(TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=item.id,
                    role="assistant"
                ))

        elif item.type == "function_call":
            # Start of a function/tool call
            # item.id is the item ID (used in deltas), item.call_id is the tool call ID
            tool_call_id = item.call_id

            # Store mapping for future deltas
            if item.id:
                self.item_id_to_tool_call_id[item.id] = tool_call_id

            events.append(ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_call_name=item.name,
                parent_message_id=None # Optional in AG_UI
            ))

    def _process_text_delta(self, event: Any, events: List[BaseEvent]):
        """Handles text content updates."""
        events.append(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=event.item_id,
            delta=event.delta
        ))

    def _process_text_done(self, event: Any, events: List[BaseEvent]):
        """Handles end of text message."""
        events.append(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=event.item_id
        ))

    def _process_function_call_delta(self, event: Any, events: List[BaseEvent]):
        """Handles tool call argument updates."""
        tool_call_id = self.item_id_to_tool_call_id.get(event.item_id)
        if tool_call_id:
            events.append(ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tool_call_id,
                delta=event.delta
            ))

    def _process_function_call_done(self, event: Any, events: List[BaseEvent]):
        """Handles end of tool call."""
        tool_call_id = self.item_id_to_tool_call_id.get(event.item_id)
        if tool_call_id:
            events.append(ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id
            ))
            # Optional: Clean up mapping if no longer needed
            # del self.item_id_to_tool_call_id[event.item_id]
