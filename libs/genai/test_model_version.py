"""Test if model_version is being accumulated during streaming"""
from google.genai import types as genai_types

# Create a mock response
response = genai_types.GenerateContentResponse(
    candidates=[
        genai_types.Candidate(
            content=genai_types.Content(parts=[genai_types.Part(text="hello")])
        )
    ]
)

# Check if model_version exists
print(f"Has model_version: {hasattr(response, 'model_version')}")
if hasattr(response, 'model_version'):
    print(f"model_version value: {response.model_version}")
    print(f"model_version type: {type(response.model_version)}")

# Try setting it
if hasattr(response, 'model_version'):
    response.model_version = "gemini-2.0-flash"
    print(f"\nAfter setting: {response.model_version}")

    # Try concatenating  (this would be the bug)
    response.model_version = response.model_version + response.model_version
    print(f"After concat: {response.model_version}")
else:
    print("\n❌ response.model_version doesn't exist in new SDK!")
    print(f"Available fields: {dir(response)}")
