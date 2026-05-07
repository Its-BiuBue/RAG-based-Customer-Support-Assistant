from typing import TypedDict
from langgraph.graph import StateGraph, END
from retriever import retrieve_info
from config import llm

# 1. Define the State
# This tracks the data as it moves through the graph
class AgentState(TypedDict):
    query: str
    context: str
    response: str
    needs_human: bool

# 2. Define the Nodes (The Actions)
def retrieve_node(state: AgentState):
    """Step 1: Retrieves context from ChromaDB based on the user's query."""
    query = state["query"]
    context = retrieve_info(query)
    return {"context": context}

def generate_node(state: AgentState):
    """Step 2: LLM tries to answer. If it lacks context, it flags for a human."""
    query = state["query"]
    context = state["context"]
    
    prompt = f"""
    You are a strictly factual Customer Support Assistant for Amazon.
    Use ONLY the following retrieved context to answer the user's question.
    
    Context: {context}
    Question: {query}
    
    Rules:
    1. If the exact answer is in the context, provide a clear, polite answer.
    2. If the context does not contain the answer, or if the question is highly complex, reply with exactly the word: ESCALATE. Do not explain why.
    """
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    if "ESCALATE" in content:
        return {"response": "Routing to human agent...", "needs_human": True}
    else:
        return {"response": content, "needs_human": False}

def human_intervention_node(state: AgentState):
    """Step 3 (Conditional): Simulates the Human-in-the-Loop (HITL) escalation."""
    print("\n" + "="*40)
    print("⚠️  HUMAN-IN-THE-LOOP TRIGGERED ⚠️")
    print(f"User asked: {state['query']}")
    print("="*40)
    # The graph pauses here to wait for human input from the terminal
    human_response = input("Agent, please type the manual response: ")
    return {"response": f"[Human Agent] {human_response}", "needs_human": False}

# 3. Define the Routing Logic
def route_after_generation(state: AgentState):
    """Decides the next edge based on the 'needs_human' flag."""
    if state["needs_human"]:
        return "human_node"
    return "end"

# 4. Build and Compile the Graph Workflow
workflow = StateGraph(AgentState)

# Add our three nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("human_node", human_intervention_node)

# Connect the nodes with edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

# This conditional edge checks if we go to a human or finish the graph
workflow.add_conditional_edges(
    "generate",
    route_after_generation,
    {
        "human_node": "human_node",
        "end": END
    }
)

workflow.add_edge("human_node", END)

# Compile into an executable application
app = workflow.compile()

# --- Testing Block ---
if __name__ == "__main__":
    print("Testing LangGraph Orchestration...\n")
    
    # Test 1: A standard question that the PDF covers
    print("--- Test 1: Standard Query ---")
    state_1 = {"query": "What is the return window for Kindle books?", "context": "", "response": "", "needs_human": False}
    result_1 = app.invoke(state_1)
    print(f"\nFinal AI Output: {result_1['response']}\n")

    # Test 2: An out-of-bounds question designed to trigger the HITL logic
    print("--- Test 2: Escalation Query ---")
    state_2 = {"query": "My Amazon delivery truck crashed into my fence, how do I get compensated?", "context": "", "response": "", "needs_human": False}
    result_2 = app.invoke(state_2)
    print(f"\nFinal Agent Output: {result_2['response']}")