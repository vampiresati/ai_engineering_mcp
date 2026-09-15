import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async def call_tool_paper_extract(paper_id: str):
    async with client:
        result = await client.call_tool("extract_info", {"paper_id": paper_id})
        print(result)

asyncio.run(call_tool_paper_extract("2609.15906v1"))
