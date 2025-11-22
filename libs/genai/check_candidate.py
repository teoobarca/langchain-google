from google.genai import types

try:
    c = types.Candidate(
        content=types.Content(
            parts=[
                types.Part(
                    text="I need to think...",
                    thought=True,
                    thought_signature=b"sig"
                )
            ]
        )
    )
    r = types.GenerateContentResponse(candidates=[c])
    print("GenerateContentResponse created successfully")
except Exception as e:
    print(f"Error: {e}")
