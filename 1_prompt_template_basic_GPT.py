# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# # PART 1: Create a ChatPromptTemplate using a template string
# template = "Give me information about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# print("-----Prompt from Template-----")
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)



# # PART 2: Prompt with Multiple Placeholders
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)



# PART 3: Prompt with System and Human Messages (Using Tuples)

# Tuples in Python are ordered, immutable collections of items.
# They are defined by enclosing elements in parentheses (), separated by commas.
# person_info = ("Alice", 30, "New York", 1.65)
# person_info[0]  = "bob"  >> wrong, tuples are immutable

# messages = [
#     ("system", "You are a coding expert proficient in {programming_language}."),
#     ("human", "Give me an example of {task}."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"programming_language": "python", "task": "summation two numbers"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)



# # Extra Informoation about Part 3.
# # This does work:
# messages = [
#     ("system", "You are a coding expert proficient in {programming_language}."),
#     HumanMessage(content="Give me an examples of summation two numbers."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"programming_language": "python"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)



messages = [
    ("system", "You are a coding expert proficient in {programming_language}."),
    HumanMessage(content="Give me an example of summation two numbers."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"programming_language": "python"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
