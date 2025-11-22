from google.genai import types
try:
    print("Blob" in dir(types))
except Exception as e:
    print(e)
