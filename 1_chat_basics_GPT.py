from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()


# OPENAI_API_KEY = "sk-proj-LKAnbJoLL2JcWMf"

# Create a ChatOpenAI model
model = ChatOpenAI(model="o4-mini-2025-04-16")

# Invoke the model with a message
result = model.invoke("what is 10 minus 7?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)



