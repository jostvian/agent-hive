from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Agent Gateway Service"}

def test_agent_endpoint_smoke():
    # Smoke test for the endpoint
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Check weather in London"
            }
        ]
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    # Check for some expected content in the SSE stream
    # Note: The JSON is escaped inside the data field, so we just check substrings
    assert "Weather in London" in response.text
