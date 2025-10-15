import os
from typing import List
from tqdm import tqdm
import yaml
import torch

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
# Uncomment if using Qdrant
# from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------
# Config Loader
# --------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------
# Vector DB Builder Class
# --------------------------
class VectorDBBuilder:
    def __init__(self, config: dict):
        # Read vector DB config
        db_cfg = config.get("vector_db", {})
        self.persist_dir = db_cfg.get("persist_dir", "embeddings/vector_db")
        self.model_name = db_cfg.get("model_name", "hkunlp/instructor-xl")
        self.batch_size = db_cfg.get("batch_size", 32)
        self.use_gpu = db_cfg.get("use_gpu", True)
        self.db_type = db_cfg.get("db_type", "chroma")
        self.collection_name = db_cfg.get("collection_name", "rag_docs")
        self.retry_attempts = db_cfg.get("retry_attempts", 3)
        self.skip_failed = db_cfg.get("skip_failed", True)

        os.makedirs(self.persist_dir, exist_ok=True)

        # Auto-detect device: MPS (Mac GPU) or CPU
        if self.use_gpu:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = "cpu"

        self.device = device
        print(f"[INFO] Using device: {self.device} for embeddings")

        # Initialize embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device}
        )

    def embed_documents(self, chunks: List[Document]) -> List[Document]:
        """Compute embeddings in batches with error handling."""
        print(f"[INFO] Computing embeddings for {len(chunks)} chunks in batches of {self.batch_size}")
        all_chunks = []
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding batches"):
            batch = chunks[i:i+self.batch_size]
            texts = [doc.page_content for doc in batch]
            attempts = 0
            while attempts < self.retry_attempts:
                try:
                    embeddings = self.embedder.embed_documents(texts)
                    for doc, emb in zip(batch, embeddings):
                        doc.metadata["_embedding"] = emb
                    all_chunks.extend(batch)
                    break
                except Exception as e:
                    attempts += 1
                    print(f"[WARN] Embedding batch failed (attempt {attempts}/{self.retry_attempts}): {e}")
                    if attempts >= self.retry_attempts:
                        if self.skip_failed:
                            print("[INFO] Skipping failed batch")
                            break
                        else:
                            raise e
        return all_chunks

    def build_vector_db(self, chunks: List[Document]):
        """Build and persist vector database."""
        chunks = self.embed_documents(chunks)

        if self.db_type.lower() == "chroma":
            print("[INFO] Building Chroma vector store...")
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedder,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name
            )
            vectordb.persist()
            print(f"[INFO] Chroma DB persisted at '{self.persist_dir}'")

        # Example for Qdrant (requires running Qdrant instance)
        # elif self.db_type.lower() == "qdrant":
        #     print("[INFO] Building Qdrant vector store...")
        #     vectordb = Qdrant.from_documents(
        #         documents=chunks,
        #         embedding=self.embedder,
        #         collection_name=self.collection_name,
        #         host="localhost",
        #         port=6333
        #     )
        #     vectordb.persist()
        #     print(f"[INFO] Qdrant DB persisted at '{self.collection_name}'")

        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

        return vectordb


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load config
    cfg = load_config()

    # Load chunks from cleaning & chunking script
    print("[INFO] Loading and chunking documents...")
    from new_clean_and_chunk import chunk_all_sources
    chunks = chunk_all_sources(config_path="config.yaml")

    # Build vector DB
    builder = VectorDBBuilder(cfg)
    vectordb = builder.build_vector_db(chunks)
    print("[INFO] Vector DB ready for RAG pipeline.")
