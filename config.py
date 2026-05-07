import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Verify the API key is loaded
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

# 1. Initialize the LLM (Gemini)
# We use gemini-1.5-flash as it is fast and ideal for orchestration and routing tasks.
# Temperature is set to 0 to keep the responses factual and deterministic for customer support.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, 
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 2. Initialize the Embedding Model (HuggingFace Local)
# This will download the open-source model directly to your machine the first time it runs.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Testing Block ---
# This block only runs if you execute this specific file directly.
if __name__ == "__main__":
    print("Testing LLM connection...")
    try:
        response = llm.invoke("Hello, are you connected?")
        print(f"LLM Response: {response.content}")
    except Exception as e:
        print(f"LLM Error: {e}")

    print("\nTesting Embedding Model...")
    try:
        vector = embeddings.embed_query("This is a test sentence.")
        print(f"Embedding generated successfully! Vector length: {len(vector)}")
    except Exception as e:
        print(f"Embedding Error: {e}")