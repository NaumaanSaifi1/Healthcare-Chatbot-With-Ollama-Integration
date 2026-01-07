import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


INPUT_FILE = "mentalhealth_train.json"
OUTPUT_FOLDER = "faiss_mentalhealth"

documents = []

print(f"Loading data from {INPUT_FILE}...")

if os.path.exists(INPUT_FILE):
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        file_content = f.read()

    try:
        data = json.loads(file_content, strict=False)
        if isinstance(data, list):
            print("Success: Detected standard JSON list.")
            for item in data:
                if 'text' in item:
                    documents.append(Document(page_content=item['text'], metadata={"source": "json_list"}))
        elif isinstance(data, dict):
             if 'text' in data:
                 documents.append(Document(page_content=data['text'], metadata={"source": "json_object"}))

    except json.JSONDecodeError:
        print("JSON list parsing failed. Attempting to parse line-by-line (JSONL)...")
        lines = file_content.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            try:
                # strict=False helps here too
                item = json.loads(line, strict=False)
                if 'text' in item:
                    documents.append(Document(page_content=item['text'], metadata={"source": f"line_{i}"}))
            except json.JSONDecodeError:
       
                try:
                    
                    cleaned_line = line.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    item = json.loads(cleaned_line, strict=False)
                    if 'text' in item:
                        documents.append(Document(page_content=item['text'], metadata={"source": f"line_{i}_cleaned"}))
                except:
                    print(f"Warning: Could not parse line {i+1}. Skipping.")
                    continue

else:
    print(f"Error: File {INPUT_FILE} not found.")
    exit()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if documents:
    print(f"Creating FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(OUTPUT_FOLDER)
    print(f"Success! Vector store saved to folder: '{OUTPUT_FOLDER}'")
else:
    print("No documents found. Please check your mentalhealth_train.json file content.")