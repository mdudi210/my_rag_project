from langchain_community.vectorstores import Chroma
# Or import from other vector DB connectors (Qdrant, etc.)
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import os

def build_db(chunks, persist_dir="embeddings/vector_db"):
    os.makedirs(persist_dir, exist_ok=True)
    # Choose your local embedding model
    embedder = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    # you can import chunks from clean_and_chunk or re-load them
    from clean_and_chunk import chunk_all_sources
    chunks = chunk_all_sources()
    db = build_db(chunks)
    print("Vector DB built and persisted.")
