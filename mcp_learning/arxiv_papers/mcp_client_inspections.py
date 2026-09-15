from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio

server_params = StdioServerParameters(
    command="uv",  # Executable
   args=[
       "run",
       "arxiv_papers/arxiv_papers_mcp_server.py"
   ],  # Command line arguments
    env=None,  # Optional environment variables
)

async def run():
    # Launch the server as a subprocess & returns the read and write streams
    # read: the stream that the client will use to read msgs from the server
    # write: the stream that client will use to write msgs to the server
    async with stdio_client(server_params) as (read, write):
        # the client session is used to initiate the connection
        # and send requests to server
        async with ClientSession(read, write) as session:
            # Initialize the connection (1:1 connection with the server)
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
#             print(tools)
            # will call the chat_loop here
            # ....
            for tool in tools.tools:
                print(f"\nTool: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Input schema: {tool.input_schema}")
            # Call a tool: this will be in the process_query method
            result = await session.call_tool("extract_info", arguments={"paper_id": "2609.15906v1"})
            print(result)


if __name__ == "__main__":
    asyncio.run(run())




# import asyncio
# from fastmcp import Client
#
# client = Client("http://localhost:8000/mcp")
#
# async def call_tool_paper_extract(paper_id: str):
#     async with client:
#         result = await client.call_tool("extract_info", {"paper_id": paper_id})
#         print(result)
#
# asyncio.run(call_tool_paper_extract("2609.15906v1"))

