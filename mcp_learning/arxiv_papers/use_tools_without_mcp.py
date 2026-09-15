from arxiv_paper_function import extract_info,search_papers
from arxiv_tool import tools
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

from ollama import chat
response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Find for paper id 2609.15922v1 directories"
        }
    ],
    tools=tools
)
# print('response')
# print(response)


if response.message.tool_calls:
    for tool_call in response.message.tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments
        print("Tool:", tool_name)
        print("Arguments:", arguments)
        function = mapping_tool_function[tool_name]
        result = function(**arguments)
#
        print("Result:", result)
