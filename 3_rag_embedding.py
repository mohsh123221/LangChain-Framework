import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
load_dotenv()  

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "langchain.txt")
db_dir = os.path.join(current_dir, "db")

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


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    print(f"\n--- Creating vector store {store_name} ---")
    Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store {store_name} ---")
   


# 1. OpenAI Embeddings
print("\n--- Using OpenAI Embeddings ---")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_vector_store(docs, openai_embeddings, "chroma_db_openai2")

# 2. Hugging Face Transformers
# Uses models from the HuggingFace library.
# Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")



# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    print(f"\n--- Querying the Vector Store {store_name} ---")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_function,)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.1},
    )
    relevant_docs = retriever.invoke(query)
    # Display the relevant results with metadata
    print(f"\n--- Relevant Documents for {store_name} ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")



# Define the user's question
query = "LangChain was launched in October 2022 as an open source project by Harrison Chase"

# Query each vector store
query_vector_store("chroma_db_openai2", query, openai_embeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)


