import pytest
import json
from app.mock_agents.weather_agent import WeatherAgent
from app.mock_agents.docs_agent import DocsAgent
from app.gateway_agent import GatewayAgent
from agent_framework import ChatMessage

def test_weather_agent():
    agent = WeatherAgent()
    response = agent.process("What is the weather in London?")
    assert response["data"]["location"] == "London"
    assert response["intent"] == "weather_report"

def test_docs_agent():
    agent = DocsAgent()
    response = agent.process("search for agent framework")
    assert response["intent"] == "doc_search_results"
    assert len(response["data"]["results"]) > 0

@pytest.mark.asyncio
async def test_gateway_agent_weather():
    agent = GatewayAgent()
    query = "Check weather in Paris"
    response = await agent.run(messages=query)
    content = json.loads(response.messages[0].text)
    assert content["type"] == "card"
    assert "Paris" in content["title"]

@pytest.mark.asyncio
async def test_gateway_agent_docs_stream():
    agent = GatewayAgent()
    query = "Search docs for agent framework"
    async for update in agent.run_stream(messages=query):
        content = json.loads(update.text)
        assert content["type"] == "card"
        assert "Results" in content["title"]
