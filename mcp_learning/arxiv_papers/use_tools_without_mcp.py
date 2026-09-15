from arxiv_paper_function import extract_info,search_papers
from arxiv_tool import tools
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}
def execute_tool(tool_name,arguments):
    function = mapping_tool_function[tool_name]
    result = function(**arguments)
    if result is None:
        result = "The operation completed but didn't return any results."
    elif isinstance(result, list):
        result = ', '.join(result)
        result = json.dumps(result, indent=2)
    else:
        result = str(result)
    return result

from ollama import chat
response = chat(model="qwen3:4b",
            messages=[{
            "role": "user",
            "content": "Find for paper id 2609.15922v1 directories"}],tools=tools)
# print('response')
# print(response)

if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments
        print("Tool:", tool_name)
        print("Arguments:", arguments)
        result=execute_tool(tool_name,arguments)
        print("Result:", result)
