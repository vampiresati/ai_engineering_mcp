import os
import subprocess

from mcp.server import MCPServer


mcp = MCPServer("Terminal")

WORKING_DIRECTORY = "/home/satvir"


@mcp.tool()
def run_command(command: str):
    """
    Execute a shell command in the specified working directory.

    Args:
        command: The terminal command to execute.

    Returns:
        A dictionary containing:
        - command: The executed command.
        - directory: The working directory.
        - return_code: The command's exit status.
        - stdout: Standard output from the command.
        - stderr: Standard error from the command.
    """

    if not os.path.isdir(WORKING_DIRECTORY):
        return {
            "command": command,
            "directory": WORKING_DIRECTORY,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Working directory does not exist: {WORKING_DIRECTORY}",
        }

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=WORKING_DIRECTORY,
    )

    return {
        "command": command,
        "directory": WORKING_DIRECTORY,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
