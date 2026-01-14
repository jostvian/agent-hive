from fastapi import FastAPI
from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint
from app.gateway_agent import GatewayAgent

app = FastAPI()

gateway_agent = GatewayAgent()

# Expose the agent via AG-UI protocol
add_agent_framework_fastapi_endpoint(
    app=app,
    agent=gateway_agent,
    path="/api/chat"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Agent Gateway Service"}
