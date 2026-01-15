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
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

def create_chunk(
    delta_content: str = None,
    delta_tool_calls: list = None,
    finish_reason: str = None,
    chunk_id: str = "msg_123"
) -> ChatCompletionChunk:
    delta_args = {}
    if delta_content is not None:
        delta_args["content"] = delta_content
    if delta_tool_calls is not None:
        delta_args["tool_calls"] = delta_tool_calls

    delta = ChoiceDelta(**delta_args)

    choice = Choice(
        delta=delta,
        finish_reason=finish_reason,
        index=0
    )

    return ChatCompletionChunk(
        id=chunk_id,
        choices=[choice],
        created=1234567890,
        model="gpt-4",
        object="chat.completion.chunk"
    )

def test_convert_text_message():
    converter = AGUIEventConverter()

    # Chunk 1: Start and content "Hello"
    chunk1 = create_chunk(delta_content="Hello")
    events1 = converter.convert_event(chunk1)

    assert len(events1) == 2
    assert isinstance(events1[0], TextMessageStartEvent)
    assert events1[0].message_id == "msg_123"
    assert isinstance(events1[1], TextMessageContentEvent)
    assert events1[1].delta == "Hello"

    # Chunk 2: Content " World"
    chunk2 = create_chunk(delta_content=" World")
    events2 = converter.convert_event(chunk2)

    assert len(events2) == 1
    assert isinstance(events2[0], TextMessageContentEvent)
    assert events2[0].delta == " World"

    # Chunk 3: Stop
    chunk3 = create_chunk(delta_content=None, finish_reason="stop")
    events3 = converter.convert_event(chunk3)

    assert len(events3) == 1
    assert isinstance(events3[0], TextMessageEndEvent)
    assert events3[0].message_id == "msg_123"

def test_convert_tool_call():
    converter = AGUIEventConverter()

    # Chunk 1: Tool call start
    tool_call_delta = ChoiceDeltaToolCall(
        index=0,
        id="call_abc",
        function=ChoiceDeltaToolCallFunction(name="get_weather", arguments="")
    )
    chunk1 = create_chunk(delta_tool_calls=[tool_call_delta])
    events1 = converter.convert_event(chunk1)

    assert len(events1) == 1
    assert isinstance(events1[0], ToolCallStartEvent)
    assert events1[0].tool_call_id == "call_abc"
    assert events1[0].tool_call_name == "get_weather"

    # Chunk 2: Tool call args "{"
    tool_call_delta2 = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(arguments="{")
    )
    chunk2 = create_chunk(delta_tool_calls=[tool_call_delta2])
    events2 = converter.convert_event(chunk2)

    assert len(events2) == 1
    assert isinstance(events2[0], ToolCallArgsEvent)
    assert events2[0].delta == "{"

    # Chunk 3: Tool call args "}"
    tool_call_delta3 = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(arguments="}")
    )
    chunk3 = create_chunk(delta_tool_calls=[tool_call_delta3])
    events3 = converter.convert_event(chunk3)

    assert len(events3) == 1
    assert isinstance(events3[0], ToolCallArgsEvent)
    assert events3[0].delta == "}"

    # Chunk 4: Finish tool calls
    chunk4 = create_chunk(finish_reason="tool_calls")
    events4 = converter.convert_event(chunk4)

    assert len(events4) == 1
    assert isinstance(events4[0], ToolCallEndEvent)
    assert events4[0].tool_call_id == "call_abc"

def test_mixed_text_and_stop():
    converter = AGUIEventConverter()

    # Chunk 1: content "Hi"
    chunk1 = create_chunk(delta_content="Hi")
    events1 = converter.convert_event(chunk1)

    assert len(events1) == 2 # Start + Content

    # Chunk 2: empty content, stop
    chunk2 = create_chunk(delta_content="", finish_reason="stop")
    events2 = converter.convert_event(chunk2)

    assert len(events2) == 1
    assert isinstance(events2[0], TextMessageEndEvent)
