🩺 MedGemma – Mental & General Health Chatbot (RAG + FastAPI)

MedGemma is a lightweight AI-powered healthcare chatbot built using LangChain, FAISS, Ollama (MedGemma), DuckDuckGo search, and FastAPI.
It provides empathetic mental health support, symptom analysis, and safe medical guidance, especially useful in remote or low-resource settings.

🚀 Features

✅ Retrieval-Augmented Generation (RAG) using FAISS

🧠 Mental health safety handling (suicide & self-harm detection)

🔍 DuckDuckGo free web search fallback

🤖 Local LLM via Ollama (MedGemma 4B)

⚡ FastAPI backend (lightweight & fast)

🌐 Frontend-ready REST API

🛡️ Safety-first medical responses

🏗️ Tech Stack
Layer	Technology
LLM	Ollama (alibayram/medgemma:4b)
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector DB	FAISS
Framework	FastAPI
RAG	LangChain
Web Search	DuckDuckGo (free, no API key)
Language	Python 3.10+
📂 Project Structure
Health chatbot 2.0/
│
├── app.py                 # FastAPI backend
├── prompt.py              # System prompt (Dr. AI)
├── faiss_mentalhealth/    # FAISS vector database
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── ingestion.py           # Dataset → FAISS (optional)

🔧 Installation
1️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\Activate      # Windows

2️⃣ Install dependencies
pip install -U fastapi uvicorn
pip install langchain langchain-community langchain-ollama
pip install langchain-huggingface faiss-cpu
pip install sentence-transformers ddgs

3️⃣ Install & start Ollama
ollama pull alibayram/medgemma:4b

📦 FAISS Database Setup

Make sure you have a FAISS folder:

faiss_mentalhealth/
├── index.faiss
├── index.pkl


If not, run your ingestion script first.

▶️ Run the Backend
uvicorn app:app --reload


Server runs at:

http://127.0.0.1:8000


Swagger Docs:

http://127.0.0.1:8000/docs

🔌 API Usage
Endpoint
POST /chat

Request
{
  "message": "I feel depressed and anxious"
}

Response
{
  "response": "I hear you, your feelings matter..."
}

🛡️ Safety System

MedGemma automatically detects:

suicide

self-harm

depression

hopelessness

If detected, it:

Shows emergency helpline info

Still provides empathetic guidance

Encourages professional help

🌍 DuckDuckGo Web Search

Used only when FAISS context is insufficient

100% free

No API key required

Improves factual accuracy

🧠 Prompt Design

The chatbot uses a Doctor-style system prompt:

Step-by-step diagnosis

OTC medicine guidance

Red-flag warnings

Compassionate tone

Remote-area assistance mindset

(Defined in prompt.py)

⚠️ Disclaimer

This chatbot is not a replacement for a licensed doctor.
It is designed for educational and emergency support purposes only.
Always consult a qualified medical professional for diagnosis and treatment.

🌱 Future Improvements

🔐 API key authentication

📱 Mobile-friendly frontend

🧾 Conversation history

📊 Medical report export

🔊 Voice input/output

👨‍💻 Author

Krish Sharma
AI / ML Engineer
Health AI | RAG | LLM Systems