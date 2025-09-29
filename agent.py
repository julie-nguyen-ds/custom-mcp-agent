import asyncio
import mlflow
import os
import json
from uuid import uuid4
from pydantic import BaseModel, create_model
from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union

from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from databricks_mcp import DatabricksOAuthClientProvider, DatabricksMCPClient
from databricks.sdk import WorkspaceClient

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.tools import BaseTool, tool

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client as connect

from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

############################################
## Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = "You are a helpful assistant that can search news article."

###############################################################################
## Configure MCP Servers for your agent
## This section sets up server connections so your agent can retrieve data or take actions.
###############################################################################

# TODO: Choose your MCP server connection type.
import nest_asyncio
nest_asyncio.apply()

# ----- Simple: Managed MCP Server (no extra setup required) -----
# Uses your Databricks Workspace settings and Personal Access Token (PAT) auth.
workspace_client = WorkspaceClient()

# Managed MCP Servers: Ready to use with default settings above
host = workspace_client.config.host
MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/functions/system/ai",
]

# ----- Advanced (optional): Custom MCP Server with OAuth -----
# For Databricks Apps hosting custom MCP servers, OAuth with a service principal is required.
# Uncomment and fill in your settings ONLY if connecting to a custom MCP server.
#
import os

workspace_client = WorkspaceClient(
    host="https://adb-4733014334286923.3.azuredatabricks.net/",
    client_id=os.getenv("DATABRICKS_CLIENT_ID"),
    client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
    auth_type="oauth-m2m",   # Enables machine-to-machine OAuth
)

# Custom MCP Servers: Add URLs below if needed (requires custom setup and OAuth above)
CUSTOM_MCP_SERVER_URLS = ["https://mcp-custom-server-test-jn-4733014334286923.3.azure.databricksapps.com/mcp"]

#####################
## MCP Tool Creation
#####################

# Define a custom LangChain tool that wraps functionality for calling MCP servers
class MCPTool(BaseTool):
    """Custom LangChain tool that wraps MCP server functionality"""

    def __init__(self, name: str, description: str, args_schema: type, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
        # Initialize the tool
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema
        )
        # Store custom attributes: MCP server URL, Databricks workspace client, and whether the tool is for a custom server
        object.__setattr__(self, 'server_url', server_url)
        object.__setattr__(self, 'workspace_client', ws)
        object.__setattr__(self, 'is_custom', is_custom)

    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        if self.is_custom:
            # Use the async method for custom MCP servers (OAuth required)
            return asyncio.run(self._run_custom_async(**kwargs))
        else:
            # Use managed MCP server via synchronous call
            mcp_client = DatabricksMCPClient(server_url=self.server_url, workspace_client=self.workspace_client)
            response = mcp_client.call_tool(self.name, kwargs)
            return "".join([c.text for c in response.content])

    async def _run_custom_async(self, **kwargs) -> str:
        """Execute custom MCP tool asynchronously"""        
        async with connect(self.server_url, auth=DatabricksOAuthClientProvider(self.workspace_client)) as (
            read_stream,
            write_stream,
            _,
        ):
            # Create an async session with the server and call the tool
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(self.name, kwargs)
                return "".join([c.text for c in response.content])

# Retrieve tool definitions from a custom MCP server (OAuth required)
async def get_custom_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a custom MCP server using OAuth"""    
    async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            return tools_response.tools

# Retrieve tool definitions from a managed MCP server
def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a managed MCP server"""
    mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
    return mcp_client.list_tools()

# Convert an MCP tool definition into a LangChain-compatible tool
def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient, is_custom: bool = False):
    """Create a LangChain tool from an MCP tool definition"""
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Map JSON schema types to Python types for input validation
    TYPE_MAPPING = {
        "integer": int,
        "number": float,
        "boolean": bool
    }
    field_definitions = {}
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        field_type = TYPE_MAPPING.get(field_type_str, str)

        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    # Dynamically create a Pydantic schema for the tool's input arguments
    args_schema = create_model(
        f"{mcp_tool.name}Args",
        **field_definitions
    )

    # Return a configured MCPTool instance
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
        is_custom=is_custom
    )

# Gather all tools from managed and custom MCP servers into a single list
async def create_mcp_tools(ws: WorkspaceClient, 
                          managed_server_urls: List[str] = None, 
                          custom_server_urls: List[str] = None) -> List[MCPTool]:
    """Create LangChain tools from both managed and custom MCP servers"""
    tools = []

    if managed_server_urls:
        # Load managed MCP tools
        for server_url in managed_server_urls:
            try:
                mcp_tools = get_managed_mcp_tools(ws, server_url)
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=False)
                    tools.append(tool)
            except Exception as e:
                print(f"Error loading tools from managed server {server_url}: {e}")

    if custom_server_urls:
        # Load custom MCP tools (async)
        for server_url in custom_server_urls:
            try:
                mcp_tools = await get_custom_mcp_tools(ws, server_url)
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=True)
                    tools.append(tool)
            except Exception as e:
                print(f"Error loading tools from custom server {server_url}: {e}")

    return tools

#####################
## Define agent logic
#####################

# The state for the agent workflow, including the conversation and any custom data
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]

# Define the LangGraph agent that can call tools
def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)  # Bind tools to the model

    # Function to check if agent should continue or finish based on last message
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If function (tool) calls are present, continue; otherwise, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # Preprocess: optionally prepend a system prompt to the conversation history
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model  # Chain the preprocessor and the model

    # The function to invoke the model within the workflow
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)  # Create the agent's state machine

    workflow.add_node("agent", RunnableLambda(call_model))  # Agent node (LLM)
    workflow.add_node("tools", ToolNode(tools))             # Tools node

    workflow.set_entry_point("agent")  # Start at agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # If the model requests a tool call, move to tools node
            "end": END,           # Otherwise, end the workflow
        },
    )
    workflow.add_edge("tools", "agent")  # After tools are called, return to agent node

    # Compile and return the tool-calling agent workflow
    return workflow.compile()

# ResponsesAgent class to wrap the compiled agent and make it compatible with Mosaic AI Responses API
class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    # Convert a LangChain message to a Responses-format dictionary
    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses output item dictionaries"""
        for message in messages:
            message = message.model_dump()  # Convert the message model to dict
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    # Return function call items for all tool calls present
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                else:
                    # Regular AI text message
                    return [
                        self.create_text_output_item(
                            text=message["content"],
                            id=message.get("id") or str(uuid4()),
                        )
                    ]
            elif role == "tool":
                # Output from tool/function execution
                return [
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                ]
            elif role == "user":
                # User messages as-is
                return [message]

    # Make a prediction (single-step) for the agent
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done" or event.type == "error"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    # Stream predictions for the agent, yielding output as it's generated
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        # Stream events from the agent graph
        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                # Stream updated messages from the workflow nodes
                for node_data in event[1].values():
                    if "messages" in node_data:
                        for item in self._langchain_to_responses(node_data["messages"]):
                            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                # Stream generated text message chunks
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except:
                    pass

# Initialize the entire agent, including MCP tools and workflow
def initialize_agent():
    """Initialize the agent with MCP tools"""
    # Create MCP tools from the configured servers
    mcp_tools = asyncio.run(create_mcp_tools(
        ws=workspace_client,
        managed_server_urls=MANAGED_MCP_SERVER_URLS,
        custom_server_urls=CUSTOM_MCP_SERVER_URLS
    ))

    # Create the agent graph with an LLM, tool set, and system prompt (if given)
    agent = create_tool_calling_agent(llm, mcp_tools, system_prompt)
    return LangGraphResponsesAgent(agent)

mlflow.langchain.autolog()
AGENT = initialize_agent()
mlflow.models.set_model(AGENT)
