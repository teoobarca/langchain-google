from google import genai
from google.genai import types

try:
    # Mock a response or check types
    print(dir(types.GenerateContentResponse))
    print(dir(types.Candidate))
    print(dir(types.Content))
    print(dir(types.Part))
except Exception as e:
    print(e)
