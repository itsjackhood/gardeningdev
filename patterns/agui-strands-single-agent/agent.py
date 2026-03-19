"""AG-UI Strands single agent with Gateway MCP tools and Code Interpreter.

Uses ag-ui-strands to produce native AG-UI SSE events.
AgentCore proxies these unchanged when deployed with --protocol AGUI.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_strands import StrandsAgent

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client

from bedrock_agentcore.identity.auth import requires_access_token
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from utils.auth import extract_user_id_from_request, setup_agentcore_context
from utils.ssm import get_ssm_parameter

logger = logging.getLogger(__name__)


# --- Gateway ---

@requires_access_token(
    provider_name=os.environ["GATEWAY_CREDENTIAL_PROVIDER_NAME"],
    auth_flow="M2M",
    scopes=[]
)
def _fetch_gateway_token(access_token: str) -> str:
    return access_token


def create_gateway_mcp_client() -> MCPClient:
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        raise ValueError("STACK_NAME environment variable is required")
    if not stack_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid STACK_NAME format")

    gateway_url = get_ssm_parameter(f"/{stack_name}/gateway_url")
    logger.info("[AGUI-STRANDS] Gateway URL: %s", gateway_url)

    return MCPClient(
        lambda: streamablehttp_client(
            url=gateway_url,
            headers={"Authorization": f"Bearer {_fetch_gateway_token()}"},
        ),
        prefix="gateway",
    )


# --- Agent creation (per-request so actor_id is per-user) ---

def create_agent(user_id: str) -> StrandsAgent:
    """Create AG-UI Strands agent with Gateway tools and Code Interpreter."""
    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0", temperature=0.1
    )

    memory_id = os.environ.get("MEMORY_ID")
    if not memory_id:
        raise ValueError("MEMORY_ID environment variable is required")

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=memory_id, session_id="default", actor_id=user_id
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    gateway_client = create_gateway_mcp_client()
    tools = [gateway_client]

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    try:
        from strands_code_interpreter import StrandsCodeInterpreterTools
        tools.append(StrandsCodeInterpreterTools(region).execute_python_securely)
        logger.info("[AGUI-STRANDS] Code Interpreter loaded")
    except Exception as e:
        logger.warning("[AGUI-STRANDS] Code Interpreter not available: %s", e)

    strands_agent = Agent(
        name="BasicAgent",
        system_prompt="You are a helpful assistant with access to tools via the Gateway and Code Interpreter. "
                      "When asked about your tools, list them and explain what they do.",
        tools=tools,
        model=bedrock_model,
        session_manager=session_manager,
    )

    return StrandsAgent(
        agent=strands_agent,
        name="agui_strands_agent",
        description="AG-UI Strands agent with Gateway MCP tools and Code Interpreter",
    )


# --- App ---

app = FastAPI(title="AG-UI Strands Agent")


@app.post("/invocations")
async def invocations(input_data: RunAgentInput, request: Request):
    try:
        setup_agentcore_context(request)
        user_id = extract_user_id_from_request(request)
        agent = create_agent(user_id)

        encoder = EventEncoder(accept=request.headers.get("accept"))

        async def event_generator():
            async for event in agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())
    except Exception:
        logger.exception("[AGUI-STRANDS] /invocations failed")
        raise


@app.get("/ping")
def ping():
    return {"status": "Healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("agent:app", host="0.0.0.0", port=port, reload=True)
