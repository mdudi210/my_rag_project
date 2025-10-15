import os
import re
import json
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def load_config(path="config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Clean unwanted text patterns like HTML remnants, whitespace, scripts."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove multiple spaces, tabs, newlines
    text = re.sub(r"\s+", " ", text)
    # Trim and normalize
    text = text.strip()
    return text


def clean_documents(docs):
    """Apply cleaning and filtering logic to all loaded documents."""
    print("[INFO] Cleaning loaded documents...")
    cleaned = []
    seen = set()
    skipped = 0

    for doc in docs:
        cleaned_text = clean_text(doc.page_content)
        if not cleaned_text or len(cleaned_text) < 50:
            skipped += 1
            continue
        if cleaned_text in seen:
            skipped += 1
            continue
        seen.add(cleaned_text)
        doc.page_content = cleaned_text
        cleaned.append(doc)

    print(f"[INFO] Cleaned {len(cleaned)} documents (skipped {skipped} empty/duplicate).")
    return cleaned


# ----------------------------------------------------------
# Chunking & Saving
# ----------------------------------------------------------

def load_documents(source_root: str):
    """Load all text documents recursively from a directory."""
    start = time.time()
    loader = DirectoryLoader(source_root, glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} raw documents in {time.time() - start:.2f}s")
    return docs


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks."""
    print(f"[INFO] Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


def save_chunks(chunks, output_root="data_sources/chunks", save_metadata=True):
    """Save chunks to disk with optional metadata tracking."""
    os.makedirs(output_root, exist_ok=True)
    metadata_records = []

    def save_file(path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            fname = f"chunk_{i}.txt"
            path = os.path.join(output_root, fname)
            metadata = {
                "file": fname,
                "source": chunk.metadata.get("source", ""),
                "start_index": chunk.metadata.get("start_index", 0),
                "length": len(chunk.page_content)
            }
            metadata_records.append(metadata)
            futures.append(executor.submit(save_file, path, chunk.page_content))

        for _ in as_completed(futures):
            pass

    if save_metadata:
        meta_path = os.path.join(output_root, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata_records, mf, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved metadata to {meta_path}")

    print(f"[INFO] Saved {len(chunks)} chunks to {output_root}")
    return metadata_records


# ----------------------------------------------------------
# Main Process
# ----------------------------------------------------------

def chunk_all_sources(config_path="config.yaml"):
    """Main entry: load config, clean data, chunk documents, and save results."""
    cfg = load_config(config_path)
    rag_cfg = cfg.get("rag", {})

    source_root = rag_cfg.get("source_root", "data_sources/web")
    output_root = rag_cfg.get("output_root", "data_sources/chunks")
    chunk_size = rag_cfg.get("chunk_size", 1000)
    chunk_overlap = rag_cfg.get("chunk_overlap", 200)

    docs = load_documents(source_root)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    save_chunks(chunks, output_root)

    print("[INFO] Cleaning & chunking process completed successfully.")
    return chunks


if __name__ == "__main__":
    chunk_all_sources()
