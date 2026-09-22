# MCP Learning

This repository contains small learning projects for building and using MCP
(Model Context Protocol) servers, tools, resources, prompts, and local clients.

The examples focus on:

- Creating MCP tools with Python
- Exposing local functions as MCP servers
- Calling MCP servers from a client
- Connecting MCP tools to local Ollama models
- Saving and reading arXiv paper metadata
- Building simple prompt and utility servers

## Project Structure

```text
.
├── arxiv_papers/
│   ├── arxiv_papers_mcp_server.py      # FastMCP arXiv research server
│   ├── arxiv_paper_function.py         # Plain Python arXiv helper functions
│   ├── arxiv_tool.py                   # Ollama function-tool schema
│   ├── use_tools_without_mcp.py        # Example using tools without MCP
│   ├── use_with_mcp.py                 # Example client using one MCP server
│   ├── mcp_using_from_server_config.py # Multi-server MCP chatbot client
│   └── server_config.json              # MCP server configuration
├── coding_prompts/
│   └── coding_prompts.py               # MCP prompt server for code review prompts
├── weather_tool/
│   └── weather_tool.py                 # MCP weather tool using wttr.in
├── terminator_tool/
│   ├── main.py                         # MCP terminal command tool
│   └── pyproject.toml
└── papers/
    └── ...                             # Saved arXiv paper metadata
```

## Requirements

- Python 3.11 or newer
- `uv` for running Python tools and MCP servers
- Node.js / `npx` for the filesystem MCP server in `server_config.json`
- Ollama for the local chatbot examples
- A local Ollama model such as `qwen2.5-coder:7b` or `qwen3:4b`

Some examples also use:

- `arxiv`
- `fastmcp`
- `mcp`
- `langchain-ollama`
- `langchain-core`
- `python-dotenv`
- `requests`

## arXiv MCP Server

The arXiv server provides tools, resources, and a prompt for research workflows.

Main file:

```bash
arxiv_papers/arxiv_papers_mcp_server.py
```

Available tools:

- `search_papers(topic, max_results=5)` searches arXiv and saves paper metadata.
- `extract_info(paper_id)` searches saved metadata for a paper ID.

Available resources:

- `papers://folders` lists saved topic folders.
- `papers://{topic}` lists saved papers for a topic.
- `papers://{topic}/{paper_id}` returns details for one saved paper.

Available prompt:

- `generate_search_prompt(topic, num_papers=5)` creates a structured research prompt.

Run the server with stdio transport:

```bash
uv run arxiv_papers/arxiv_papers_mcp_server.py
```

Search results are saved under:

```text
papers/<topic>/papers_info.json
```

## MCP Chatbot Client

The multi-server chatbot client loads MCP servers from:

```bash
arxiv_papers/server_config.json
```

Configured servers include:

- `fetch`
- `filesystem`
- `arxiv_papers`

Run the chatbot:

```bash
uv run arxiv_papers/mcp_using_from_server_config.py
```

The client uses Ollama through LangChain and expects the model:

```text
qwen2.5-coder:7b
```

You can change the model in `MCP_ChatBot.__init__`.

## Plain arXiv Function Example

You can also test the arXiv search functionality without running an MCP server:

```bash
uv run arxiv_papers/arxiv_paper_function.py
```

The non-MCP tool-calling example is:

```bash
uv run arxiv_papers/use_tools_without_mcp.py
```

## Weather Tool

The weather tool exposes `get_weather(city)` as an MCP tool and uses `wttr.in`.

Run it with:

```bash
uv run weather_tool/weather_tool.py
```

Example returned data:

```json
{
  "result": "found",
  "wind": "12",
  "temparature": "28",
  "wind unit": "km/h",
  "temparature unit": "celsius"
}
```

## Coding Prompts Server

The coding prompt server exposes MCP prompts for reviewing Angular and release
changes by ticket number.

Run it with:

```bash
uv run coding_prompts/coding_prompts.py
```

It loads environment values from:

```text
coding_prompts/.prompts_env
```

Expected values:

```env
URL=...
PROJECT=...
COMPANY=...
```

## Terminal Tool

The terminal tool exposes a `run_command(command)` MCP tool.

Run it from its folder:

```bash
cd terminator_tool
uv run main.py
```

By default, commands run inside:

```text
/home/satvir
```

Change `WORKING_DIRECTORY` in `terminator_tool/main.py` if you want to target a
different directory.

## Notes

- The arXiv FastMCP server currently includes a Keycloak auth provider configured
  for local development at `http://localhost:8080/realms/mcp-realm`.
- The examples are learning-oriented and may need dependency installation before
  first use.
- Generated paper data is stored in `papers/` and can be inspected directly as
  JSON.
