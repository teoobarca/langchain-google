"""Test that tools can be properly serialized"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def test_tool(query: str) -> str:
    """A test tool."""
    return f"Result for: {query}"

# Create LLM with tool
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="fake-key-for-testing"
)

# Try to prep a request (this will test the serialization)
try:
    request = llm._prepare_request(
        messages=[HumanMessage(content="test")],
        tools=[test_tool]
    )
    print("✅ Tool serialization successful!")
    print(f"Config tools type: {type(request['config'].tools)}")
    if request['config'].tools:
        print(f"First tool type: {type(request['config'].tools[0])}")
        print(f"First tool keys: {list(request['config'].tools[0].keys()) if isinstance(request['config'].tools[0], dict) else 'Not a dict'}")
except Exception as e:
    print(f"❌ Tool serialization failed: {e}")
    import traceback
    traceback.print_exc()
