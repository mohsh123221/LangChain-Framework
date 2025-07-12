import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "langchain.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_openai") 

# Create Ollama embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

# Read the text content from the file
loader = TextLoader(file_path)   
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Create the vector store and persist it automatically
print("\n--- Creating vector store ---")
Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
print("\n--- Finished creating vector store ---")