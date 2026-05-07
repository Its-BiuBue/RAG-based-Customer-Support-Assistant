import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import embeddings

# Define where the local database will be saved
CHROMA_PATH = "chroma_db"

def process_and_store_document(pdf_path: str):
    """Loads a PDF, chunks the text, and stores embeddings in ChromaDB."""
    print(f"Loading document: {pdf_path}")
    
    # 1. Load the PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None
        
    # 2. Chunk the text
    # We use RecursiveCharacterTextSplitter to ensure paragraphs and sentences stay together
    # chunk_size=500 and overlap=50 is a good baseline for customer support queries
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split document into {len(chunks)} chunks.")
    
    # 3. Save to Vector Database (ChromaDB)
    print("Saving chunks to ChromaDB. This might take a moment...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Success! Vector database saved to local folder: {CHROMA_PATH}")
    return db

# --- Testing Block ---
if __name__ == "__main__":
    # Here is exactly where the file name is defined!
    sample_pdf = "knowledge_base/support_policy.pdf"
    
    if not os.path.exists(sample_pdf):
        print(f"Setup required: Please place your PDF named 'support_policy.pdf' inside the '{os.path.dirname(sample_pdf)}' folder.")
    else:
        process_and_store_document(sample_pdf)