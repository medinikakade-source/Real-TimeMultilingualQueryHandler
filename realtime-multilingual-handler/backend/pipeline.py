# backend/pipeline.py
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Updated, future-proof imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_texts(data_dir="data"):
    p = Path(data_dir)
    docs = []
    for f in p.glob("**/*.txt"):
        text = f.read_text(encoding="utf-8")
        docs.append({"page_content": text, "metadata": {"source": str(f)}})
    return docs

def build_vectorstore(data_dir="data", persist_directory="chroma_db",
                      model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                      collection_name="kb_collection"):
    print("Loading documents...")
    docs = load_texts(data_dir)
    if not docs:
        raise RuntimeError("No documents found in data/ — add a .txt file and try again.")

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = []
    for d in docs:
        chunks = splitter.split_text(d["page_content"])
        for i, c in enumerate(chunks):
            split_docs.append(Document(page_content=c, metadata={"source": d["metadata"]["source"], "chunk": i}))

    print(f"Created {len(split_docs)} chunks. Creating embeddings (this may take a moment)...")
    emb = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=emb,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vectordb.persist()
    print("Vector store created and persisted at:", persist_directory)
    return vectordb

def query_vectorstore(query_text, k=4, persist_directory="chroma_db",
                      model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                      collection_name="kb_collection"):
    emb = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=emb, collection_name=collection_name)
    results = vectordb.similarity_search_with_score(query_text, k=k)
    return results

if __name__ == "__main__":
    build_vectorstore()
    print("Done. Use query_vectorstore(query_text) to test retrieval.")