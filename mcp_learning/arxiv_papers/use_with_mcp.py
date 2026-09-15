
import asyncio
import json
from typing import List

from ollama import chat
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCP_ChatBot:

    def __init__(self):

        self.session = None

        self.available_tools: List[dict] = []

        # Local Qwen model
        self.model = "qwen2.5-coder:7b"


    # ============================================================
    # PROCESS QUERY
    # ============================================================

    async def process_query(self, query):

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        while True:

            # ----------------------------------------------------
            # Ask Qwen
            # ----------------------------------------------------

            response = chat(
                model=self.model,
                messages=messages,
                tools=self.available_tools
            )

            assistant_message = response["message"]

            print("\nDEBUG - Ollama response:")
            print(response)

            print("\nDEBUG - Assistant message:")
            print(assistant_message)


            # ----------------------------------------------------
            # Add assistant response
            # ----------------------------------------------------

            messages.append(assistant_message)


            # ====================================================
            # 1. NATIVE OLLAMA TOOL CALL
            # ====================================================

            tool_calls = assistant_message.get("tool_calls")


            if tool_calls:

                print(
                    "\nNative tool call detected."
                )

                await self.execute_tools(
                    tool_calls,
                    messages
                )

                continue


            # ====================================================
            # 2. QWEN GENERATED JSON TOOL CALL
            # ====================================================

            content = assistant_message.get(
                "content",
                ""
            ).strip()


            tool_call = self.parse_json_tool_call(
                content
            )


            if tool_call:

                print(
                    "\nJSON tool call detected."
                )

                print(
                    f"Tool: {tool_call['name']}"
                )

                print(
                    f"Arguments: {tool_call['arguments']}"
                )


                # Convert to Ollama-like tool call
                tool_calls = [
                    {
                        "function": tool_call
                    }
                ]


                await self.execute_tools(
                    tool_calls,
                    messages
                )


                continue


            # ====================================================
            # 3. NORMAL FINAL ANSWER
            # ====================================================

            print("\nQwen:")
            print(content)

            break


    # ============================================================
    # PARSE JSON TOOL CALL
    # ============================================================

    def parse_json_tool_call(self, content):

        if not content:
            return None


        # --------------------------------------------------------
        # Try normal JSON
        # --------------------------------------------------------

        try:

            data = json.loads(content)

        except json.JSONDecodeError:

            return None


        # --------------------------------------------------------
        # Check expected structure
        # --------------------------------------------------------

        if not isinstance(data, dict):

            return None


        if "name" not in data:

            return None


        if "arguments" not in data:

            return None


        tool_name = data["name"]

        arguments = data["arguments"]


        # --------------------------------------------------------
        # Make sure this is actually one of our MCP tools
        # --------------------------------------------------------

        available_tool_names = {
            tool["function"]["name"]
            for tool in self.available_tools
        }


        if tool_name not in available_tool_names:

            return None


        return {
            "name": tool_name,
            "arguments": arguments
        }


    # ============================================================
    # EXECUTE MCP TOOLS
    # ============================================================

    async def execute_tools(
        self,
        tool_calls,
        messages
    ):

        for tool_call in tool_calls:

            function = tool_call["function"]


            tool_name = function["name"]


            tool_args = function.get(
                "arguments",
                {}
            )


            print(
                f"\nCalling MCP tool: {tool_name}"
            )


            print(
                f"Arguments: {tool_args}"
            )


            # ----------------------------------------------------
            # Call MCP server
            # ----------------------------------------------------

            result = await self.session.call_tool(
                tool_name,
                arguments=tool_args
            )


            # ----------------------------------------------------
            # Extract text from MCP result
            # ----------------------------------------------------

            result_text_parts = []


            for content in result.content:

                if hasattr(content, "text"):

                    result_text_parts.append(
                        content.text
                    )


            result_text = "\n".join(
                result_text_parts
            )


            print(
                "\nMCP Tool Result:"
            )


            print(result_text)


            # ----------------------------------------------------
            # Send tool result back to Qwen
            # ----------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": result_text
                }
            )


    # ============================================================
    # CHAT LOOP
    # ============================================================

    async def chat_loop(self):

        print(
            "\n========================================"
        )

        print(
            "      MCP + Qwen Chatbot Started"
        )

        print(
            "========================================"
        )


        print(
            f"\nModel: {self.model}"
        )


        print(
            "\nAvailable MCP tools:"
        )


        for tool in self.available_tools:

            print(
                f"  - {tool['function']['name']}"
            )


        print(
            "\nType 'quit' to exit."
        )


        while True:

            query = input(
                "\nQuery: "
            ).strip()


            if query.lower() == "quit":

                print(
                    "\nGoodbye!"
                )

                break


            if not query:

                continue


            try:

                await self.process_query(
                    query
                )

            except Exception as e:

                print(
                    f"\nError: {type(e).__name__}: {e}"
                )


    # ============================================================
    # CONNECT TO MCP SERVER
    # ============================================================

    async def connect_to_server_and_run(self):

        server_params = StdioServerParameters(

            command="uv",

            args=[
                "run",
                "arxiv_papers/arxiv_papers_mcp_server.py"
            ],

            env=None
        )


        print(
            "\nStarting MCP server..."
        )


        # --------------------------------------------------------
        # Start MCP server
        # --------------------------------------------------------

        async with stdio_client(
            server_params
        ) as (read, write):


            # ----------------------------------------------------
            # MCP session
            # ----------------------------------------------------

            async with ClientSession(
                read,
                write
            ) as session:


                self.session = session


                # ------------------------------------------------
                # Initialize MCP
                # ------------------------------------------------

                print(
                    "\nInitializing MCP connection..."
                )


                await session.initialize()


                # ------------------------------------------------
                # Discover tools
                # ------------------------------------------------

                response = await session.list_tools()


                tools = response.tools


                print(
                    "\nConnected to MCP server!"
                )


                print(
                    "\nMCP tools:"
                )


                for tool in tools:

                    print(
                        f"  - {tool.name}"
                    )


                # ------------------------------------------------
                # Convert MCP → Ollama
                # ------------------------------------------------

                self.available_tools = []


                for tool in tools:

                    ollama_tool = {

                        "type": "function",

                        "function": {

                            "name": tool.name,

                            "description": (
                                tool.description
                                or ""
                            ),

                            "parameters": (
                                tool.input_schema
                            )
                        }
                    }


                    self.available_tools.append(
                        ollama_tool
                    )


                print(
                    "\nOllama tool schemas:"
                )


                for tool in self.available_tools:

                    print(tool)


                # ------------------------------------------------
                # Start chatbot
                # ------------------------------------------------

                await self.chat_loop()


# ================================================================
# MAIN
# ================================================================

async def main():

    chatbot = MCP_ChatBot()

    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":

    asyncio.run(main())

