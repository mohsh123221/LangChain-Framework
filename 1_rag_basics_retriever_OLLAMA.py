import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_ollama")

# Create Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# load chroma vector store
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "what is LangChain?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")