from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 10 minus 4?"),
]


# ---- LangChain OpenAI Chat Model Example ----

# Create a ChatOpenAI model
model = ChatOpenAI(model="o4-mini-2025-04-16")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


result = model.invoke(messages)
print(f"Answer from Google: {result.content}")



# ---- OLLAMA Chat Model Example ----

# Create a OLLAMA model https://ollama.com/library?sort=newest
model = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1")

result = model.invoke(messages)
print(f"Answer from OLLAMA : {result}")