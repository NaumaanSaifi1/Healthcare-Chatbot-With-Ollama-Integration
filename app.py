import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing system_prompt, fallback if missing
try:
    from prompt import system_prompt
except ImportError:
    logger.warning("prompt.py not found. Using default prompt.")
    system_prompt = "Context: {context} \n Query: {query} \n Answer:"

DB_FOLDER = "faiss_mentalhealth"
K_RESULTS = 3
OLLAMA_MODEL = "alibayram/medgemma:4b" 
SAFETY_KEYWORDS = [
    "suicide", "depression", "self-harm", "hopeless",
    "kill myself", "die", "end it", "want to disappear"
]


resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load resources on startup 
    and clean them up on shutdown.
    """
    logger.info("Loading resources...")
    
    
    resources["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    if os.path.exists(DB_FOLDER):
        resources["vectorstore"] = FAISS.load_local(
            DB_FOLDER, 
            resources["embeddings"], 
            allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded successfully.")
    else:
        logger.warning(f"Database folder '{DB_FOLDER}' not found. Retrieval will be skipped.")
        resources["vectorstore"] = None


    resources["llm"] = OllamaLLM(model=OLLAMA_MODEL, temperature=0.7)


    try:
        resources["duck_tool"] = DuckDuckGoSearchRun()
    except Exception as e:
        logger.warning(f"DuckDuckGo tool failed to load: {e}")
        resources["duck_tool"] = None
    
    yield

    resources.clear()
    logger.info("Resources cleared.")

app = FastAPI(title="MedGemma Chatbot API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

def check_safety(query: str):
    return any(k in query.lower() for k in SAFETY_KEYWORDS)

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    response_text = ""
    
  
    if check_safety(query):
        response_text += (
            "**I hear you, and your feelings matter.** Please contact a trained professional immediately.\n\n"
            "- **India:** Call 112 or 108\n"
            "- **International:** [Find A Helpline](https://findahelpline.com)\n\n"
        )

  
    vectorstore = resources.get("vectorstore")
    duck_tool = resources.get("duck_tool")
    llm = resources.get("llm")

    
    context_text = ""
    if vectorstore:
        try:
            results = vectorstore.similarity_search(query, k=K_RESULTS)
            context_text = "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

   
    if not context_text.strip() and duck_tool:
        try:
            logger.info("Falling back to DuckDuckGo search...")
            context_text = duck_tool.run(query)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            context_text = "No additional info found online."

    
    formatted_prompt = system_prompt.format(context=context_text, query=query)
    
    try:
        llm_response = llm.invoke(formatted_prompt)
    except Exception as e:
        llm_response = f"I encountered an error processing your request: {str(e)}"

    response_text += str(llm_response)
    
    return {"response": response_text}