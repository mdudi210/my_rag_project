import os
import json
from datetime import datetime
import yaml
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# --------------------------
# Load config
# --------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
chat_cfg = cfg["chat"]
vector_cfg = cfg["vector_db"]

HISTORY_FILE = chat_cfg.get("history_file", "chat_history.json")
MAX_HISTORY_EXCHANGES = chat_cfg.get("max_history_exchanges", 10)


# --------------------------
# Chat History Utilities
# --------------------------
def load_chat_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_chat_history(history: List[dict]):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def append_chat(history: List[dict], role: str, message: str):
    history.append({
        "role": role,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    # Limit chat memory
    if len(history) > MAX_HISTORY_EXCHANGES * 2:
        history = history[-MAX_HISTORY_EXCHANGES*2:]
    save_chat_history(history)
    return history


# --------------------------
# Query RAG with general knowledge fallback
# --------------------------
def query_local_rag(query: str):
    # Load vector store
    embedding_model = HuggingFaceEmbeddings(model_name=vector_cfg.get("model_name", "hkunlp/instructor-xl"))
    vectordb = Chroma(
        persist_directory=vector_cfg.get("persist_dir", "embeddings/vector_db"),
        embedding_function=embedding_model
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": vector_cfg.get("k", 3)})

    # Initialize LLM
    llm = Ollama(model=vector_cfg.get("llm_model", "llama3"))

    # Load previous chat
    chat_history = load_chat_history()
    context_text = ""
    if chat_history:
        context_text = "\n".join([f"{c['role']}: {c['message']}" for c in chat_history])

    # Combine query with previous chat context
    query_with_context = f"{context_text}\nUser: {query}" if context_text else query

    # Create RAG QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Run query
    answer = qa.run(query_with_context)

    # Optionally, include sources from retrieved docs
    sources = retriever.get_relevant_documents(query)
    source_info = "\n".join([f"- {doc.metadata.get('source', 'unknown')}" for doc in sources])
    if source_info:
        answer += f"\n\n[Sources]\n{source_info}"

    # Append to chat history
    chat_history = append_chat(chat_history, "User", query)
    chat_history = append_chat(chat_history, "AI", answer)

    return answer


# --------------------------
# Interactive Chat
# --------------------------
if __name__ == "__main__":
    print("[INFO] Starting local RAG chat. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        response = query_local_rag(user_input)
        print(f"AI: {response}")
