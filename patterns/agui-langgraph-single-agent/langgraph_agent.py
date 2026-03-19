"""AG-UI LangGraph single agent with Gateway MCP tools and Memory.

Uses ag-ui-langgraph to produce native AG-UI SSE events.
AgentCore proxies these unchanged when deployed with --protocol AGUI.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph import LangGraphAgent

from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from bedrock_agentcore.identity.auth import requires_access_token
from utils.auth import extract_user_id_from_request, setup_agentcore_context
from utils.ssm import get_ssm_parameter
from patched_langgraph_agent import PatchedLangGraphAgent

logger = logging.getLogger(__name__)


@requires_access_token(
    provider_name=os.environ["GATEWAY_CREDENTIAL_PROVIDER_NAME"],
    auth_flow="M2M",
    scopes=[]
)
async def _fetch_gateway_token(access_token: str) -> str:
    return access_token


async def create_gateway_mcp_client() -> MultiServerMCPClient:
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        raise ValueError("STACK_NAME environment variable is required")
    if not stack_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid STACK_NAME format")

    gateway_url = get_ssm_parameter(f"/{stack_name}/gateway_url")
    logger.info("[AGUI-LG] Gateway URL: %s", gateway_url)

    fresh_token = await _fetch_gateway_token()
    logger.info("[AGUI-LG] Gateway token fetched (%d chars)", len(fresh_token))
    return MultiServerMCPClient({
        "gateway": {
            "transport": "streamable_http",
            "url": gateway_url,
            "headers": {"Authorization": f"Bearer {fresh_token}"}
        }
    })


async def create_agent(user_id: str) -> LangGraphAgent:
    system_prompt = """You are a helpful assistant with access to tools via the Gateway.
    When asked about your tools, list them and explain what they do."""

    bedrock_model = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.1,
        streaming=True
    )

    memory_id = os.environ.get("MEMORY_ID")
    if not memory_id:
        raise ValueError("MEMORY_ID environment variable is required")

    checkpointer = AgentCoreMemorySaver(
        memory_id=memory_id,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    mcp_client = await create_gateway_mcp_client()
    tools = await mcp_client.get_tools()
    logger.info("[AGUI-LG] Loaded %d tools from Gateway", len(tools))

    # Code Interpreter
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    try:
        from langgraph_code_interpreter import LangGraphCodeInterpreterTools
        code_tools = LangGraphCodeInterpreterTools(region)
        tools.append(code_tools.execute_python_securely)
        logger.info("[AGUI-LG] Code Interpreter loaded")
    except Exception as e:
        logger.warning("[AGUI-LG] Code Interpreter not available: %s", e)

    graph = create_react_agent(
        model=bedrock_model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=system_prompt
    )

    return PatchedLangGraphAgent(
        name="agui_langgraph_agent",
        graph=graph,
        description="AG-UI LangGraph agent with Gateway MCP tools and Memory",
        config={"configurable": {"actor_id": user_id}},
    )


# --- Create app and register endpoint ---

app = FastAPI(title="AG-UI LangGraph Agent")


@app.post("/invocations")
async def invocations(input_data: RunAgentInput, request: Request):
    try:
        setup_agentcore_context(request)
        user_id = extract_user_id_from_request(request)
        agent = await create_agent(user_id)

        encoder = EventEncoder(accept=request.headers.get("accept"))

        async def event_generator():
            async for event in agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(event_generator(), media_type=encoder.get_content_type())
    except Exception:
        logger.exception("[AGUI-LG] /invocations failed")
        raise


@app.get("/ping")
def ping():
    return {"status": "Healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("langgraph_agent:app", host="0.0.0.0", port=port, reload=True)
