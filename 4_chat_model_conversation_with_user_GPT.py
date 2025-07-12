from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv() 



# Create a ChatOpenAI model
model = ChatOpenAI(model="o4-mini-2025-04-16")

# Use a list to store messages
chat_history = []  

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
# Add system message to chat history
chat_history.append(system_message)  

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)
