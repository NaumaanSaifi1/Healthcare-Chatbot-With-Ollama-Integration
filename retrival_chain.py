import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DB_FOLDER = "faiss_mentalhealth"

def main():
    # 1. Check if database exists
    if not os.path.exists(DB_FOLDER):
        print(f"Error: Database folder '{DB_FOLDER}' not found. Run the analysis script first.")
        return

    # 2. Load the same embedding model used for training
    print("Loading model... (this may take a second)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Load the local Vector Store
    print("Loading vector database...")
    try:
        vectorstore = FAISS.load_local(
            DB_FOLDER, 
            embeddings, 
            allow_dangerous_deserialization=True # Necessary for local files
        )
        print("Chatbot is ready! Type 'exit' to stop.\n")
    except Exception as e:
        print(f"Failed to load database: {e}")
        return

    # 4. Chat Loop
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Search the database for the 2 most similar entries
        results = vectorstore.similarity_search(query, k=2)

        if results:
            # We get the 'page_content' which contains the full conversation
            best_match = results[0].page_content
            
            # Simple parsing to separate Human and Assistant for cleaner display
            # (Assumes your data follows the <HUMAN>...<ASSISTANT> format)
            if "<ASSISTANT>:" in best_match:
                answer = best_match.split("<ASSISTANT>:")[-1].strip()
            else:
                answer = best_match

            print(f"\nAssistant: {answer}")
            
            # Optional: Show the source metadata to prove it came from your file
            # print(f"[Source: {results[0].metadata}]")
        else:
            print("\nAssistant: I'm sorry, I don't have information on that topic in my knowledge base.")

if __name__ == "__main__":
    main()