from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_ollama.llms import OllamaLLM

# Load environment variables from .env
load_dotenv()

# Create a OllamaLLM model
model = OllamaLLM(model="hf.co/mradermacher/DAPO-Coding-Qwen2.5-1.5B-Instruct-GGUF:Q8_0")


# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a coding expert proficient in {programming_language}."),
        ("human", "give me just one example of function to {task}."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x)) # Format the input into a prompt
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages())) # Invoke the model with the formatted prompt
parse_output = RunnableLambda(lambda x: x.upper())  # Convert output to uppercase

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
result = chain.invoke({"programming_language": "python", "task": "subtract two numbers"})

# Output
print(result)



