from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def query_local_rag(query, persist_dir="embeddings/vector_db"):
    embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query)

if __name__ == "__main__":
    q = input("Enter your question: ")
    resp = query_local_rag(q)
    print("Answer:")
    print(resp)
