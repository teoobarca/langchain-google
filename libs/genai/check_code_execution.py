from google.genai import types
try:
    print("CodeExecution" in dir(types))
except Exception as e:
    print(e)
