"""Test tool schema transformation"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import json

@tool
def test_tool(query: str) -> str:
    """A test tool."""
    return f"Result for: {query}"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="fake-key-for-testing"
)

try:
    request = llm._prepare_request(
        messages=[HumanMessage(content="test")],
        tools=[test_tool]
    )
    print("✅ Tool transformation successful!")
    print(f"\nFirst tool structure:")
    print(json.dumps(request['config'].tools[0], indent=2))

    # Check for old SDK fields that should be removed
    tool_str = json.dumps(request['config'].tools[0])
    if 'type_' in tool_str:
        print("\n❌ ERROR: Old SDK 'type_' field still present!")
    elif 'format_' in tool_str:
        print("\n❌ ERROR: Old SDK 'format_' field still present!")
    else:
        print("\n✅ No old SDK fields detected")

except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
