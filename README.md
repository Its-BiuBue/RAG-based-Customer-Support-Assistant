# RAG-Based Customer Support Assistant (with LangGraph & HITL)

This project is an enterprise-grade AI Customer Support Assistant built using Retrieval-Augmented Generation (RAG) and stateful orchestration. It processes proprietary company policies to answer customer queries accurately and uses a graph-based workflow to escalate complex or out-of-bounds questions to a human agent via a Human-in-the-Loop (HITL) mechanism.

## 🚀 Key Features
* **Agentic Routing:** Uses a Large Language Model (LLM) to intelligently route questions based on confidence and context availability.
* **Semantic Search:** Employs local HuggingFace embeddings and ChromaDB to perform highly accurate vector similarity searches across ingested PDF documents.
* **Human-in-the-Loop (HITL):** Built with LangGraph to freeze the application state and hand off complex queries to a human agent, preventing AI hallucinations.
* **Zero-Latency Local Database:** Vectors and embeddings are handled 100% locally to reduce latency and API costs.

## 🛠️ Technology Stack
* **Orchestration:** LangGraph & LangChain
* **LLM Reasoning Engine:** Google Gemini (gemini-2.5-flash)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`
* **Vector Database:** ChromaDB
* **Document Processing:** PyPDFLoader & RecursiveCharacterTextSplitter
* **Interface:** Command Line Interface (CLI)

---

## 📂 Project Structure

```text
├── knowledge_base/          # Directory containing the source PDFs (e.g., support_policy.pdf)
├── chroma_db/               # Auto-generated local vector database
├── docs/                    # Architecture diagrams and design documents (HLD/LLD)
├── .env                     # Environment variables (API Keys)
├── requirements.txt         # Python dependencies
├── config.py                # Initializes the LLM and Embedding models
├── document_processor.py    # Loads the PDF, chunks text, and saves to ChromaDB
├── retriever.py             # Executes similarity searches against the vector database
├── graph_workflow.py        # Defines the LangGraph state machine, nodes, and HITL logic
└── app.py                   # The main CLI application loop
```
## Setup & Installation Instructions
Follow these steps to run the project locally on your machine.
* **Clone the Repository**

Download or clone this project folder to your local machine and open it in your preferred IDE (like VS Code).

* **Set Up a Virtual Environment**

It is recommended to isolate your dependencies. Open your terminal and run:
```
python -m venv rag_env
```

Activate the environment:

Windows: .\rag_env\Scripts\activate

Mac/Linux: source rag_env/bin/activate

* **Install Dependencies**
Install all required libraries using the provided requirements file:
```
pip install -r requirements.txt
```

*(Note: If you don't have a requirements.txt, you can install manually via: pip install langchain langgraph langchain-community chromadb pypdf sentence-transformers langchain-google-genai langchain-huggingface python-dotenv)*

* **Configure API Keys**
```
Get a free API key from Google AI Studio.

Create a file named .env in the root directory.

Add your key to the file:

GOOGLE_API_KEY="your_actual_api_key_here"
```

* **Prepare the Knowledge Base**

Create a folder named knowledge_base.

Place a PDF document inside it named support_policy.pdf (e.g., an Amazon Return Policy or company FAQ).

## ▶️ How to Run the Project
Step 1: Ingest the Data
Before asking questions, you must convert your PDF into searchable vectors. Run the document processor:
```
python document_processor.py
```

Expected Output: The script will load the PDF, chunk it, generate embeddings (downloading the HuggingFace model on the first run), and save them to a new chroma_db folder.

Step 2: Start the Support Assistant

Once the database is populated, launch the interactive Command Line Interface:

```
python app.py
```
Step 3: Test the Application

You can now chat with the bot! Try two types of prompts to test the routing logic:

Standard Query: Ask a question covered in your PDF (e.g., "What is the return window for defective items?"). The AI will answer instantly.

Escalation Query: Ask a complex, out-of-bounds question (e.g., "I need to sue the company, who do I talk to?"). The system will print ⚠️ HUMAN-IN-THE-LOOP TRIGGERED ⚠️ and pause the terminal, waiting for you (the human agent) to type a manual response.


**To quit the loop**

Type : 
```
quit
```
