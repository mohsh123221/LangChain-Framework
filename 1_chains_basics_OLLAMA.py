from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.llms import OllamaLLM

# Create a OLLAMA model
model = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a coding expert proficient in {programming_language}."),
        ("human", "give me just one example to {task}."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
#chain = prompt_template | model 
chain = prompt_template | model  | StrOutputParser()

# Run the chain
result = chain.invoke({"programming_language": "python", "task": "subtract two numbers"})

# Output
print(result)
