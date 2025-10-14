from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_all_sources(source_root="data_sources", output_root="data_sources/chunks"):
    loader = DirectoryLoader(source_root, glob="**/*.txt")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    # Optionally: save each chunk as a .txt with metadata
    os.makedirs(output_root, exist_ok=True)
    for i, chunk in enumerate(chunks):
        fname = f"chunk_{i}.txt"
        with open(os.path.join(output_root, fname), "w", encoding="utf-8") as f:
            f.write(chunk.page_content)
    return chunks
