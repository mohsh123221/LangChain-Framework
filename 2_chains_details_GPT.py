from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="o4-mini-2025-04-16")


# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a coding expert proficient in {programming_language}."),
        ("human", "give me just one example of function to {task}."),
    ]
)


# Create individual runnables (steps in the chain)
prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content.upper()) 

# # Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=prompt, middle=[invoke_model], last=parse_output)
# chain = prompt | invoke_model | parse_output

# Run the chain
result = chain.invoke({"programming_language": "python", "task": "subtract two numbers"})

# Output
print(result)
