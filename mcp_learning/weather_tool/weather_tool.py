import requests
import os
import subprocess
import json
from mcp.server import MCPServer
mcp = MCPServer("Weather")
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather information for a city using wttr.in.
    Args:
        city: Name of the city or location.
    Returns:
        Weather information as plain text.
    """
    noresult={"result": "not found ","wind":0,"temparature":0,"wind unit":"km/h","temparature unit":"celsius"}
    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=%t,+%w",
            timeout=10
        )
        r=response.text
    except Exception as e:
        return json.dumps(noresult)
    r=r.split(',')
    temp = r[0].replace("+", "").replace("°C", "")
    wind = r[1].replace("→", "").replace("km/h", "")
    r={"result": "found","wind":wind,"temparature":temp,"wind unit":"km/h","temparature unit":"celsius"}
    return json.dumps(r)

if __name__ == "__main__":
    mcp.run(transport="stdio")
#     print(get_weather("delhi"))
