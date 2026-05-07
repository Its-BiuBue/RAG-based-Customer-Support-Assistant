from graph_workflow import app, AgentState

def main():
    print("=======================================================")
    print("🤖 Welcome to the Amazon RAG Support Assistant")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=======================================================\n")

    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Ending session. Goodbye!")
            break
            
        if not user_input.strip():
            continue

        # Initialize the state for this query
        state = {"query": user_input, "context": "", "response": "", "needs_human": False}
        
        try:
            # Run the graph
            result = app.invoke(state)
            
            # The HITL print statements are handled inside the graph_workflow.py
            # Here we just print the final resolved answer
            print(f"\nAssistant: {result['response']}\n")
            print("-" * 55)
            
        except Exception as e:
            print(f"\n[System Error]: {e}\n")

if __name__ == "__main__":
    main()