import pytest
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
# We need to mock the OpenAI events since constructing them might be verbose or require validation
from openai.types.responses import (
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

# Mocking constructors or using simple objects since Pydantic models might require all fields
# But since we have the libraries installed, we can try to use them if possible,
# or use duck-typing/mock objects if constructors are too strict.

def test_convert_text_message():
    converter = AGUIEventConverter()

    # 1. Output Item Added (Message)
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

    events1 = converter.convert_event(event1)
    assert len(events1) == 1
    assert isinstance(events1[0], TextMessageStartEvent)
    assert events1[0].message_id == "msg_123"

    # 2. Text Delta "Hello"
    event2 = ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="msg_123",
        delta="Hello",
        output_index=0,
        content_index=0,
        logprobs=[],
        sequence_number=1
    )

    events2 = converter.convert_event(event2)
    assert len(events2) == 1
    assert isinstance(events2[0], TextMessageContentEvent)
    assert events2[0].delta == "Hello"

    # 3. Text Done
    event3 = ResponseTextDoneEvent(
        type="response.output_text.done",
        item_id="msg_123",
        output_index=0,
        content_index=0,
        sequence_number=2,
        text="Hello",
        logprobs=[]
    )

    events3 = converter.convert_event(event3)
    assert len(events3) == 1
    assert isinstance(events3[0], TextMessageEndEvent)
    assert events3[0].message_id == "msg_123"

def test_convert_tool_call():
    converter = AGUIEventConverter()

    # 1. Output Item Added (Function Call)
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

    events1 = converter.convert_event(event1)
    assert len(events1) == 1
    assert isinstance(events1[0], ToolCallStartEvent)
    assert events1[0].tool_call_id == "call_abc123"
    assert events1[0].tool_call_name == "get_weather"

    # 2. Args Delta "{"
    event2 = ResponseFunctionCallArgumentsDeltaEvent(
        type="response.function_call_arguments.delta",
        item_id="item_tool_1",
        delta="{",
        output_index=0,
        sequence_number=1
    )

    events2 = converter.convert_event(event2)
    assert len(events2) == 1
    assert isinstance(events2[0], ToolCallArgsEvent)
    assert events2[0].tool_call_id == "call_abc123" # Should be mapped from item_id
    assert events2[0].delta == "{"

    # 3. Args Done
    event3 = ResponseFunctionCallArgumentsDoneEvent(
        type="response.function_call_arguments.done",
        item_id="item_tool_1",
        output_index=0,
        sequence_number=2,
        arguments="{}",
        name="get_weather"
    )

    events3 = converter.convert_event(event3)
    assert len(events3) == 1
    assert isinstance(events3[0], ToolCallEndEvent)
    assert events3[0].tool_call_id == "call_abc123"
