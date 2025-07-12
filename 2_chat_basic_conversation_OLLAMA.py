from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM


# Create a OLLAMA model
model = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math expressions, output must be numbers only without any text"),
    HumanMessage(content="What is 10 minus 4?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI : {result}")


# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math equations, output must be numbers only"),
    HumanMessage(content="What is 10 minus 4?"),
    AIMessage(content="6"),
    HumanMessage(content="What is 10 times 5?"),
    
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI : {result}")
