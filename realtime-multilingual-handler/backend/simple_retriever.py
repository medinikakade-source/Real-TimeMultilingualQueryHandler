# backend/simple_retriever.py
from pathlib import Path
import numpy as np
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use the langchain-huggingface class if installed; otherwise langchain_community is OK.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_chunk(data_dir="data", chunk_size=500, chunk_overlap=100):
    p = Path(data_dir)
    docs = []
    for f in p.glob("**/*.txt"):
        txt = f.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(txt)
        for i, c in enumerate(chunks):
            docs.append({"text": c, "source": str(f), "chunk": i})
    return docs

def embed_texts(texts: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    emb = HuggingFaceEmbeddings(model_name=model_name)
    # Langchain embeddings return lists of floats; convert to np arrays
    vectors = [np.array(v, dtype=np.float32) for v in emb.embed_documents(texts)]
    return vectors

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # return cosine similarity
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def build_index(data_dir="data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    docs = load_and_chunk(data_dir)
    texts = [d["text"] for d in docs]
    if not texts:
        return [], [], []
    vectors = embed_texts(texts, model_name=model_name)
    return docs, texts, vectors

def query_index(query: str, docs, vectors, model_name="sentence-transformers/all-MiniLM-L6-v2", k=5):
    emb = HuggingFaceEmbeddings(model_name=model_name)
    qv = np.array(emb.embed_query(query), dtype=np.float32)
    scores = [cosine_sim(qv, v) for v in vectors]
    # pair (score, doc) and sort descending
    pairs = sorted([(s, docs[i]) for i, s in enumerate(scores)], key=lambda x: x[0], reverse=True)
    return pairs[:k]

if __name__ == "__main__":
    docs, texts, vectors = build_index()
    print("Indexed chunks:", len(docs))
    if not docs:
        print("No docs found in data/. Add a sample.txt and run again.")
        raise SystemExit(0)
    q = "How does the multilingual handler work?"
    print("Query:", q)
    res = query_index(q, docs, vectors, k=5)
    for i,(score, doc) in enumerate(res):
        print(f"\nResult {i} | score={score:.4f} | source={doc['source']} | chunk={doc['chunk']}")
        print(doc['text'][:400].replace("\n"," "))
