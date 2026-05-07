from langchain_community.vectorstores import Chroma
from config import embeddings

# Define where the local database is saved
CHROMA_PATH = "chroma_db"

def retrieve_info(query: str) -> str:
    """Searches the vector database for the most relevant text chunks."""
    
    # 1. Connect to the existing local Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 2. Perform a similarity search
    # k=3 means we want the top 3 most relevant chunks from those 30 chunks you created
    results = db.similarity_search(query, k=3)
    
    # 3. Combine the results into a single readable string for the LLM to read
    context = "\n\n".join([doc.page_content for doc in results])
    
    return context

# --- Testing Block ---
if __name__ == "__main__":
    print("Testing Retriever...")
    
    # Let's test with a specific question from the Amazon policy you pasted
    test_query = "What is the return policy for Alexa Paid Skills?"
    
    print(f"User Query: {test_query}\n")
    print("Retrieving context from ChromaDB...\n")
    
    retrieved_context = retrieve_info(test_query)
    
    print("--- Retrieved Context ---")
    print(retrieved_context)
    print("-------------------------")