import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from prompt import system_prompt  # Your Dr. AI prompt

# --- Configuration ---
DB_FOLDER = "faiss_mentalhealth"
K_RESULTS = 3
OLLAMA_MODEL = "alibayram/medgemma:4b"
SAFETY_KEYWORDS = [
    "suicide", "depression", "self-harm", "hopeless",
    "kill myself", "die", "end it", "want to disappear"
]

# Initialize DuckDuckGo tool
duck_tool = DuckDuckGoSearchRun()

def stream_print(text, delay=0.02):
    """Print text like an animation, char by char."""
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

def main():
    # 1. Validate DB
    if not os.path.exists(DB_FOLDER):
        print(f"Error: Database folder '{DB_FOLDER}' not found. Please run ingestion first.")
        return

    # 2. Load Resources
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading knowledge base...")
    try:
        vectorstore = FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Critical Error loading database: {e}")
        return

    print(f"Connecting to AI model: {OLLAMA_MODEL}...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.7)

    print("\n✅ Mental Health & Health Chatbot Ready! (Type 'exit' to quit)\n")

    # 3. Main Loop
    while True:
        try:
            query = input("User: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye. Stay safe!")
                break

            # Safety Filter
            if any(k in query.lower() for k in SAFETY_KEYWORDS):
                stream_print("\nAssistant: I hear you, your feelings matter. Please contact a trained professional immediately.")
                stream_print("- India: Call 112 or 108 (Emergency Services)")
                stream_print("- International: Find hotlines at https://findahelpline.com")

            # Retrieval
            results = vectorstore.similarity_search(query, k=K_RESULTS)
            context_text = "\n\n".join([doc.page_content for doc in results]) if results else ""

            # DuckDuckGo fallback if no context
            if not context_text.strip():
                stream_print("\n[Fetching online information...]")
                context_text = duck_tool.run(query)

            # LLM Prompt
            formatted_prompt = system_prompt.format(context=context_text, query=query)
            response = llm.invoke(formatted_prompt)

            # Streamed response
            stream_print(f"\nAssistant: {response.strip()}\n", delay=0.01)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye. Stay safe!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
