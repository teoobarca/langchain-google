"""Debug tool serialization"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def test_tool(query: str) -> str:
    """A test tool."""
    return f"Result for: {query}"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="fake-key-for-testing"
)

# Get the formatted tools before serialization
from langchain_google_genai._function_utils import convert_to_genai_function_declarations
formatted_tools_before = [convert_to_genai_function_declarations([test_tool])]

print("Formatted tools BEFORE serialization:")
print(f"Type: {type(formatted_tools_before[0])}")
print(f"Has to_dict: {hasattr(formatted_tools_before[0], 'to_dict')}")
print(f"Has model_dump: {hasattr(formatted_tools_before[0], 'model_dump')}")

# Now test the actual serialization in _prepare_request
try:
    request = llm._prepare_request(
        messages=[HumanMessage(content="test")],
        tools=[test_tool]
    )
    print("\n✅ Request prepared successfully!")
    print(f"\nConfig tools type: {type(request['config'].tools)}")
    if request['config'].tools:
        print(f"First tool type: {type(request['config'].tools[0])}")
        # Try to use tool_to_dict manually
        from langchain_google_genai._function_utils import tool_to_dict
        manual_dict = tool_to_dict(formatted_tools_before[0])
        print(f"\nManual tool_to_dict result type: {type(manual_dict)}")
        print(f"Manual tool_to_dict keys: {list(manual_dict.keys())}")
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
