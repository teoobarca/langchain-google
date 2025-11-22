from google.genai import types
try:
    print(types.Tool.__annotations__)
except Exception as e:
    print(e)
