from google import genai
from google.genai import types
import inspect

print("Checking google.genai.types for thinking related configs...")

# Check GenerateContentConfig or similar
try:
    # In the new SDK, it's often types.GenerateContentConfig
    # Let's list fields of types.ThinkingConfig if it exists

    if hasattr(types, "ThinkingConfig"):
        print("\nThinkingConfig found.")
        # It might be a Pydantic model or a TypedDict or a Proto wrapper
        # Let's inspect it
        tc = types.ThinkingConfig
        print("Type:", tc)
        try:
            # If pydantic
            print("Fields:", tc.model_fields.keys())
        except:
            print("Dir:", dir(tc))
    else:
        print("\nThinkingConfig NOT found in types.")

    # Check GenerateContentConfig
    if hasattr(types, "GenerateContentConfig"):
        print("\nGenerateContentConfig found.")
        gcc = types.GenerateContentConfig
        try:
            print("Fields:", gcc.model_fields.keys())
        except:
            pass

except Exception as e:
    print(e)
