from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from ocr_tool_pytesseract import ocr_read_document_pytesseract

tools = [ocr_read_document_pytesseract]
llm = ChatOllama(model="qwen3:4b",temperature=0)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a receipt analysis assistant.

You have access to an OCR tool.

When the user asks you to analyze a receipt:

1. Use the OCR tool to read the image.
2. Identify all item prices.
3. Calculate the subtotal.
4. Find the tax.
5. Calculate the expected total.
6. Compare it with the printed total.
7. Report whether the total is correct.

Do not invent values.

If OCR cannot read something, clearly say so.
"""
)


# ============================================================
# 5. Task
# ============================================================

task = """
Please process the document at 'receipt.jpg' and evaluate the correctness
of the total.

Use the OCR tool.

Check:

- Individual item amounts
- Total
- Cash
- Calculated total
- Printed total
- Difference
Tell me whether the receipt total is correct.
"""


response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": task
        }
    ]
})


print("\n" + "=" * 80)
print("QWEN 3:4B ANALYSIS")
print("=" * 80)

print(response["messages"][-1].content)
