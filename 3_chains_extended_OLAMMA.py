from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_ollama.llms import OllamaLLM


# Create a OllamaLLM model
model = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a coding expert proficient in {programming_language}."),
        ("human", "give me just one Brief example to {function}."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"programming_language": "python", "function": "subtract two numbers"})

# Output
print(result)
