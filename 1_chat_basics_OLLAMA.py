from langchain_ollama.llms import OllamaLLM

# Create a OLLAMA model
model = OllamaLLM(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:custom1')

# Invoke the model with a message
result = model.invoke("what is 10 minus 7?")
print("AI Response: \n" + result)



