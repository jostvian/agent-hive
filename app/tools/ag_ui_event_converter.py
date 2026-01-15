from typing import List, Optional, Dict, Set, Any, AsyncIterable
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
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)

class AGUIEventConverter:
    """
    A generic converter to transform OpenAI Responses API streaming events (ResponseStreamEvent)
    into AG_UI events.
    """
    def __init__(self):
        # Maps item_id (from OpenAI) to tool_call_id (for AG_UI/OpenAI correlation)
        self.item_id_to_tool_call_id: Dict[str, str] = {}

    async def convert_event(self, stream: AsyncIterable[ResponseStreamEvent]) -> AsyncIterable[BaseEvent]:
        """
        Converts a stream of OpenAI ResponseStreamEvents into a stream of AG_UI BaseEvents.
        """
        async for event in stream:
            # We collect events in a list internally for the single step, then yield them
            events_to_yield: List[BaseEvent] = []

            match event:
                case ResponseOutputItemAddedEvent():
                    self._process_output_item_added(event, events_to_yield)

                case ResponseTextDeltaEvent():
                    self._process_text_delta(event, events_to_yield)

                case ResponseTextDoneEvent():
                    self._process_text_done(event, events_to_yield)

                case ResponseFunctionCallArgumentsDeltaEvent():
                    self._process_function_call_delta(event, events_to_yield)

                case ResponseFunctionCallArgumentsDoneEvent():
                    self._process_function_call_done(event, events_to_yield)

                case _:
                    # Ignore other events
                    pass

            for ag_ui_event in events_to_yield:
                yield ag_ui_event

    def _process_output_item_added(self, event: ResponseOutputItemAddedEvent, events: List[BaseEvent]):
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

    def _process_text_delta(self, event: ResponseTextDeltaEvent, events: List[BaseEvent]):
        """Handles text content updates."""
        events.append(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=event.item_id,
            delta=event.delta
        ))

    def _process_text_done(self, event: ResponseTextDoneEvent, events: List[BaseEvent]):
        """Handles end of text message."""
        events.append(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=event.item_id
        ))

    def _process_function_call_delta(self, event: ResponseFunctionCallArgumentsDeltaEvent, events: List[BaseEvent]):
        """Handles tool call argument updates."""
        tool_call_id = self.item_id_to_tool_call_id.get(event.item_id)
        if tool_call_id:
            events.append(ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tool_call_id,
                delta=event.delta
            ))

    def _process_function_call_done(self, event: ResponseFunctionCallArgumentsDoneEvent, events: List[BaseEvent]):
        """Handles end of tool call."""
        tool_call_id = self.item_id_to_tool_call_id.get(event.item_id)
        if tool_call_id:
            events.append(ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id
            ))
