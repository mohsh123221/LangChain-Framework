from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="o4-mini-2025-04-16")




# PART 1: Create a ChatPromptTemplate using a template string
print("-----Prompt from Template-----")
template = "Tell me some facts about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)




# PART 2: Prompt with Multiple Placeholders
print("\n----- Prompt with Multiple Placeholders -----\n")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "monkey"})

result = model.invoke(prompt)
print(result.content)





# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a coding expert proficient in {programming_language}."),
    ("human", "give me an example to {task}."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"programming_language": "python", "task": "summation two numbers"})

result = model.invoke(prompt)
print(result.content)
