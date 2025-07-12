from langchain_ollama.llms import OllamaLLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Create a OLLAMA model
model = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1", temperature=1, max_tokens=100)


chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)
