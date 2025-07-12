from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
# Load environment variables from .env file
load_dotenv()

def search_tivaly(query):
    """Searches tavily and returns the summary of the first result."""
    from tavily import TavilyClient

    try:
      
      client = TavilyClient(api_key="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")
      return client.search(query=query)
    except:
        return "I couldn't find any information on that."
    

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=3)
    except:
        return "I couldn't find any information on that."


# Define the tools that the agent can use
tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
    Tool(
        name="search tivaly",
        func=search_tivaly,
        description="Useful for when you need to know new or updated information about a topic.",
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")
llm = ChatOpenAI(model="gpt-4.1-2025-04-14")

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: tivaly and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
