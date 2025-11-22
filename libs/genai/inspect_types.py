from google.genai import types
try:
    print("FunctionDeclaration" in dir(types))
    print(dir(types))
except Exception as e:
    print(e)
