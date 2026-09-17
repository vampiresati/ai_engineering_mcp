from mcp.server.mcpserver import MCPServer
from dotenv import load_dotenv
import os

mcp = MCPServer("Codingprompts")

load_dotenv("/home/satvir/ai_engineering/mcp_learning/coding_prompts/.prompts_env")

URL = os.getenv("URL", "")
PROJECT = os.getenv("PROJECT", "")
COMPANY = os.getenv("COMPANY", "")


@mcp.prompt()
def review_angular_code(ticket_number: str) -> str:
    """Generate a prompt to review angular changes related to a ticket"""
    return (
        f"commits of {ticket_number} into angular what is done in last merge of this, "
        f"i need screen to screen display with url {URL} with project {PROJECT} "
        f"and company id {COMPANY}, just of screen to screen display changes. "
        f"do not open browser and do not run tests"
    )


@mcp.prompt()
def review_release_code(ticket_number: str) -> str:
    """Generate a prompt to review release/release changes related to a ticket"""
    return (
        f"commits of {ticket_number} into release/release what is done in last merge of this, "
        f"i need screen to screen display with url {URL} with project {PROJECT} "
        f"and company id {COMPANY}, just of screen to screen display changes. "
        f"do not open browser and do not run tests"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
#     print(get_weather("delhi"))
