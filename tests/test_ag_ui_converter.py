import pytest
from typing import AsyncIterable
from app.tools.ag_ui_event_converter import AGUIEventConverter
from ag_ui.core.events import (
    EventType,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from openai.types.responses import (
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseStreamEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

async def async_iter(items: list) -> AsyncIterable:
    for item in items:
        yield item

@pytest.mark.asyncio
async def test_convert_text_message():
    converter = AGUIEventConverter()

    # Events
    msg_item = ResponseOutputMessage(
        id="msg_123",
        type="message",
        role="assistant",
        content=[],
        status="in_progress"
    )
    event1 = ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        item=msg_item,
        output_index=0,
        sequence_number=0
    )

    event2 = ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="msg_123",
        delta="Hello",
        output_index=0,
        content_index=0,
        logprobs=[],
        sequence_number=1
    )

    event3 = ResponseTextDoneEvent(
        type="response.output_text.done",
        item_id="msg_123",
        output_index=0,
        content_index=0,
        sequence_number=2,
        text="Hello",
        logprobs=[]
    )

    stream = async_iter([event1, event2, event3])

    events = []
    async for event in converter.convert_event(stream):
        events.append(event)

    assert len(events) == 3
    assert isinstance(events[0], TextMessageStartEvent)
    assert events[0].message_id == "msg_123"

    assert isinstance(events[1], TextMessageContentEvent)
    assert events[1].delta == "Hello"

    assert isinstance(events[2], TextMessageEndEvent)
    assert events[2].message_id == "msg_123"

@pytest.mark.asyncio
async def test_convert_tool_call():
    converter = AGUIEventConverter()

    # Events
    tool_item = ResponseFunctionToolCall(
        id="item_tool_1",
        call_id="call_abc123",
        type="function_call",
        name="get_weather",
        arguments="",
        status="in_progress"
    )
    event1 = ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        item=tool_item,
        output_index=0,
        sequence_number=0
    )

    event2 = ResponseFunctionCallArgumentsDeltaEvent(
        type="response.function_call_arguments.delta",
        item_id="item_tool_1",
        delta="{",
        output_index=0,
        sequence_number=1
    )

    event3 = ResponseFunctionCallArgumentsDoneEvent(
        type="response.function_call_arguments.done",
        item_id="item_tool_1",
        output_index=0,
        sequence_number=2,
        arguments="{}",
        name="get_weather"
    )

    stream = async_iter([event1, event2, event3])

    events = []
    async for event in converter.convert_event(stream):
        events.append(event)

    assert len(events) == 3
    assert isinstance(events[0], ToolCallStartEvent)
    assert events[0].tool_call_id == "call_abc123"
    assert events[0].tool_call_name == "get_weather"

    assert isinstance(events[1], ToolCallArgsEvent)
    assert events[1].tool_call_id == "call_abc123"
    assert events[1].delta == "{"

    assert isinstance(events[2], ToolCallEndEvent)
    assert events[2].tool_call_id == "call_abc123"
