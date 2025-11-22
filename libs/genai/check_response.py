from google.genai import types
try:
    print(dir(types.GenerateContentResponse))
except Exception as e:
    print(e)
