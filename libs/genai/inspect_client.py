from google import genai
import inspect

try:
    client = genai.Client(api_key="test")
    print(dir(client))
except Exception as e:
    print(e)
