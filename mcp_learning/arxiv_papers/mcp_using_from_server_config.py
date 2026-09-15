import asyncio
import json
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pydantic import BaseModel, create_model


# ============================================================
# Load environment
# ============================================================

load_dotenv()


# ============================================================
# MCP Tool Definition
# ============================================================

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


# ============================================================
# MCP ChatBot
# ============================================================

class MCP_ChatBot:

    def __init__(self):

        # ----------------------------------------------------
        # MCP sessions
        # ----------------------------------------------------

        self.sessions: List[ClientSession] = []

        # ----------------------------------------------------
        # Async cleanup manager
        # ----------------------------------------------------

        self.exit_stack = AsyncExitStack()

        # ----------------------------------------------------
        # Local Ollama model
        # ----------------------------------------------------

        self.llm = ChatOllama(
            model="qwen2.5-coder:7b",
            temperature=0,
        )

        # ----------------------------------------------------
        # All MCP tools exposed to LangChain
        # ----------------------------------------------------

        self.available_tools: List[StructuredTool] = []

        # ----------------------------------------------------
        # Mapping:
        #
        # MCP tool name
        #       ↓
        # MCP session
        #
        # Example:
        #
        # search_papers → arxiv session
        # read_file    → filesystem session
        # fetch        → fetch session
        # ----------------------------------------------------

        self.tool_to_session: Dict[str, ClientSession] = {}

        # ----------------------------------------------------
        # Store original MCP schemas
        # ----------------------------------------------------

        self.tool_schemas: Dict[str, dict] = {}

    # ========================================================
    # Convert JSON Schema → Python Type
    # ========================================================

    def json_type_to_python(
        self,
        property_schema: dict
    ):

        property_type = property_schema.get(
            "type",
            "string"
        )

        if property_type == "string":
            return str

        elif property_type == "integer":
            return int

        elif property_type == "number":
            return float

        elif property_type == "boolean":
            return bool

        elif property_type == "array":
            return list

        elif property_type == "object":
            return dict

        return Any

    # ========================================================
    # Create Dynamic Pydantic Arguments Schema
    # ========================================================

    def create_args_schema(
        self,
        tool_name: str,
        input_schema: dict
    ):

        properties = input_schema.get(
            "properties",
            {}
        )

        required = input_schema.get(
            "required",
            []
        )

        fields = {}

        # ----------------------------------------------------
        # Convert MCP JSON schema into Pydantic fields
        # ----------------------------------------------------

        for property_name, property_schema in properties.items():

            python_type = self.json_type_to_python(
                property_schema
            )

            description = property_schema.get(
                "description",
                ""
            )

            # Required argument
            if property_name in required:

                fields[property_name] = (
                    python_type,
                    ...
                )

            # Optional argument
            else:

                fields[property_name] = (
                    python_type | None,
                    None
                )

        # ----------------------------------------------------
        # No parameters
        # ----------------------------------------------------

        if not fields:

            class EmptyArgs(BaseModel):
                pass

            return EmptyArgs

        # ----------------------------------------------------
        # Dynamic Pydantic model
        # ----------------------------------------------------

        ArgsModel = create_model(
            f"{tool_name}_Args",
            **fields
        )

        return ArgsModel

    # ========================================================
    # Connect to ONE MCP Server
    # ========================================================

    async def connect_to_server(
        self,
        server_name: str,
        server_config: dict
    ) -> None:

        print(
            f"\nConnecting to MCP server: "
            f"{server_name}"
        )

        try:

            # ------------------------------------------------
            # Create MCP server parameters
            # ------------------------------------------------

            server_params = StdioServerParameters(
                **server_config
            )

            # ------------------------------------------------
            # Start stdio transport
            # ------------------------------------------------

            stdio_transport = (
                await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            )

            read, write = stdio_transport

            # ------------------------------------------------
            # Create MCP session
            # ------------------------------------------------

            session = (
                await self.exit_stack.enter_async_context(
                    ClientSession(
                        read,
                        write
                    )
                )
            )

            # ------------------------------------------------
            # Initialize MCP
            # ------------------------------------------------

            await session.initialize()

            self.sessions.append(session)

            # ------------------------------------------------
            # Get tools from MCP server
            # ------------------------------------------------

            response = await session.list_tools()

            tools = response.tools

            print(
                f"Connected to {server_name}"
            )

            print(
                "Tools:",
                [tool.name for tool in tools]
            )

            # ------------------------------------------------
            # Register every MCP tool
            # ------------------------------------------------

            for mcp_tool in tools:

                tool_name = mcp_tool.name

                tool_description = (
                    mcp_tool.description
                    or f"MCP tool: {tool_name}"
                )

                input_schema = (
                    getattr(
                        mcp_tool,
                        "input_schema",
                        None
                    )
                    or getattr(
                        mcp_tool,
                        "inputSchema",
                        None
                    )
                    or {}
                )

                # --------------------------------------------
                # Save session mapping
                # --------------------------------------------

                self.tool_to_session[
                    tool_name
                ] = session

                # --------------------------------------------
                # Save schema
                # --------------------------------------------

                self.tool_schemas[
                    tool_name
                ] = input_schema

                # --------------------------------------------
                # Create Pydantic args schema
                # --------------------------------------------

                args_schema = self.create_args_schema(
                    tool_name,
                    input_schema
                )

                # --------------------------------------------
                # Create async wrapper
                # --------------------------------------------

                async def call_mcp_tool(
                    _tool_name=tool_name,
                    _session=session,
                    **kwargs
                ):

                    print(
                        f"\n[MCP] Calling: "
                        f"{_tool_name}"
                    )

                    print(
                        f"[MCP] Arguments: "
                        f"{kwargs}"
                    )

                    result = await _session.call_tool(
                        _tool_name,
                        arguments=kwargs
                    )

                    return self.format_mcp_result(
                        result
                    )

                # --------------------------------------------
                # Convert MCP tool into LangChain tool
                # --------------------------------------------

                langchain_tool = StructuredTool.from_function(
                    coroutine=call_mcp_tool,
                    name=tool_name,
                    description=tool_description,
                    args_schema=args_schema,
                )

                self.available_tools.append(
                    langchain_tool
                )

        except Exception as e:

            print(
                f"\nFailed to connect to "
                f"{server_name}:"
            )

            print(
                repr(e)
            )

    # ========================================================
    # Format MCP Result
    # ========================================================

    def format_mcp_result(
        self,
        result
    ) -> str:

        try:

            contents = result.content

            output = []

            for content in contents:

                # --------------------------------------------
                # Text content
                # --------------------------------------------

                if hasattr(content, "text"):

                    output.append(
                        content.text
                    )

                # --------------------------------------------
                # Other MCP content
                # --------------------------------------------

                else:

                    output.append(
                        str(content)
                    )

            if output:

                return "\n".join(output)

            return str(result)

        except Exception:

            return str(result)

    # ========================================================
    # Connect to ALL MCP Servers
    # ========================================================

    async def connect_to_servers(self):

        print(
            "\nLoading MCP configuration..."
        )

        try:

            # ------------------------------------------------
            # Read configuration
            # ------------------------------------------------

            config_path = (
                Path(__file__).resolve().parent
                / "server_config.json"
            )

            with open(
                config_path,
                "r"
            ) as file:

                data = json.load(file)

            servers = data.get(
                "mcpServers",
                {}
            )

            if not servers:

                raise ValueError(
                    "No MCP servers found "
                    "in server_config.json"
                )

            print(
                f"Found {len(servers)} MCP servers."
            )

            # ------------------------------------------------
            # Connect to every server
            # ------------------------------------------------

            for server_name, server_config in (
                servers.items()
            ):

                await self.connect_to_server(
                    server_name,
                    server_config
                )

            # ------------------------------------------------
            # Print final tool list
            # ------------------------------------------------

            print(
                "\n================================"
            )

            print(
                "All MCP tools available:"
            )

            for tool in self.available_tools:

                print(
                    f"  - {tool.name}"
                )

            print(
                "================================"
            )

        except Exception as e:

            print(
                "\nError loading "
                "server configuration:"
            )

            print(
                repr(e)
            )

            raise

    # ========================================================
    # Process User Query
    # ========================================================

    async def process_query(
        self,
        query: str
    ):

        # ----------------------------------------------------
        # Initial conversation
        # ----------------------------------------------------

        messages = [

            SystemMessage(
                content="""
You are an AI assistant running locally using
Ollama and Qwen.

You are connected to multiple MCP servers.

Available MCP capabilities may include:

1. Web fetching
2. Local filesystem access
3. arXiv paper search
4. arXiv paper information extraction

IMPORTANT:

- Use MCP tools when they are required.
- Do not invent tool results.
- If a tool can provide the information, use the tool.
- After receiving a tool result, analyze it and answer
  the user's question.
"""
            ),

            HumanMessage(
                content=query
            ),
        ]

        # ----------------------------------------------------
        # Bind all MCP tools to Qwen
        # ----------------------------------------------------

        llm_with_tools = self.llm.bind_tools(
            self.available_tools
        )

        # ----------------------------------------------------
        # Tool calling loop
        # ----------------------------------------------------

        while True:

            print(
                "\n[Qwen] Thinking..."
            )

            response = await llm_with_tools.ainvoke(
                messages
            )

            # ------------------------------------------------
            # Add Qwen response
            # ------------------------------------------------

            messages.append(
                response
            )

            # ------------------------------------------------
            # No tool call
            # ------------------------------------------------

            if not response.tool_calls:

                print(
                    "\nQwen:"
                )

                print(
                    response.content
                )

                break

            # ------------------------------------------------
            # Qwen requested tools
            # ------------------------------------------------

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call.get(
                    "args",
                    {}
                )

                tool_call_id = tool_call["id"]

                print(
                    "\n--------------------------------"
                )

                print(
                    "Tool requested:"
                )

                print(
                    f"Name: {tool_name}"
                )

                print(
                    f"Arguments: {tool_args}"
                )

                print(
                    "--------------------------------"
                )

                # --------------------------------------------
                # Find MCP session
                # --------------------------------------------

                session = (
                    self.tool_to_session.get(
                        tool_name
                    )
                )

                if session is None:

                    error_message = (
                        f"Tool '{tool_name}' "
                        f"was not found."
                    )

                    print(
                        error_message
                    )

                    messages.append(
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tool_call_id
                        )
                    )

                    continue

                # --------------------------------------------
                # Execute MCP tool
                # --------------------------------------------

                try:

                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_args
                    )

                    result_text = (
                        self.format_mcp_result(
                            result
                        )
                    )

                    print(
                        "\nTool result:"
                    )

                    print(
                        result_text
                    )

                    # ----------------------------------------
                    # Send result back to Qwen
                    # ----------------------------------------

                    messages.append(
                        ToolMessage(
                            content=result_text,
                            tool_call_id=tool_call_id
                        )
                    )

                except Exception as e:

                    error_message = (
                        f"Error executing tool "
                        f"'{tool_name}': "
                        f"{str(e)}"
                    )

                    print(
                        error_message
                    )

                    messages.append(
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tool_call_id
                        )
                    )

            # ------------------------------------------------
            # Continue loop
            #
            # Qwen receives tool results and can:
            #
            # 1. Call another tool
            # 2. Call multiple tools
            # 3. Give final answer
            # ------------------------------------------------

    # ========================================================
    # Interactive Chat Loop
    # ========================================================

    async def chat_loop(self):

        print(
            "\n========================================"
        )

        print(
            "       Ollama MCP Chatbot"
        )

        print(
            "========================================"
        )

        print(
            "Model: qwen2.5-coder:7b"
        )

        print(
            "MCP Servers:"
        )

        print(
            "  - fetch"
        )

        print(
            "  - filesystem"
        )

        print(
            "  - arxiv_papers"
        )

        print(
            "\nType your query."
        )

        print(
            "Type 'quit' to exit."
        )

        print(
            "========================================"
        )

        while True:

            try:

                query = input(
                    "\nQuery: "
                ).strip()

                if not query:
                    continue

                if query.lower() == "quit":

                    print(
                        "\nExiting..."
                    )

                    break

                await self.process_query(
                    query
                )

            except (KeyboardInterrupt, EOFError):

                print(
                    "\n\nExiting..."
                )

                break

            except Exception as e:

                print(
                    "\nError:"
                )

                print(
                    repr(e)
                )

    # ========================================================
    # Cleanup
    # ========================================================

    async def cleanup(self):

        print(
            "\nClosing MCP connections..."
        )

        await self.exit_stack.aclose()

        print(
            "MCP connections closed."
        )


# ============================================================
# Main
# ============================================================

async def main():

    chatbot = MCP_ChatBot()

    try:

        # ----------------------------------------------------
        # Connect all MCP servers
        # ----------------------------------------------------

        await chatbot.connect_to_servers()

        # ----------------------------------------------------
        # Show registered tools
        # ----------------------------------------------------

        print(
            "\n========== REGISTERED TOOLS =========="
        )

        for tool in chatbot.available_tools:

            print(
                "\nNAME:",
                tool.name
            )

            print(
                "DESCRIPTION:",
                tool.description
            )

            print(
                "ARGS:",
                tool.args
            )

        print(
            "\n======================================="
        )

        # ----------------------------------------------------
        # Start chatbot
        # ----------------------------------------------------

        await chatbot.chat_loop()

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        await chatbot.cleanup()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
