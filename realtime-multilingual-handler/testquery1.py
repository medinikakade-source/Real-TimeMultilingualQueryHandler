from backend.pipeline import build_vectorstore, query_vectorstore, load_texts
from pathlib import Path
import os, sys, json

print("CWD:", os.getcwd())
print("Data files:", os.listdir("data") if os.path.exists("data") else "data missing")

# Show raw file contents length
for f in Path("data").glob("*.txt"):
    print(f" - {f} size={f.stat().st_size}")

# Rebuild the vectorstore and capture chunk count
try:
    # We'll call build_vectorstore but also re-implement the chunking step to print counts for transparency
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    docs = load_texts("data")
    print("Loaded docs count:", len(docs))
    total_chunks = 0
    for d in docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(d["page_content"])
        print(f"  file={d['metadata']['source']} -> chunks={len(chunks)}")
        total_chunks += len(chunks)
    print("Total chunks to index:", total_chunks)
except Exception as e:
    print("Error during pre-check chunking:", repr(e))

# Now (re)build the vectordb (this will persist to chroma_db)
try:
    print("\nRunning build_vectorstore() ...")
    build_vectorstore()
    print("build_vectorstore() finished.")
except Exception as e:
    print("build_vectorstore() failed:", repr(e))
    sys.exit(1)

# Inspect the chroma_db collection size using the same embeddings so we don't rely on magic:
try:
    print("\nRunning a query and printing up to 10 results...")
    res = query_vectorstore("How does the multilingual handler work?", k=10)
    print("Query returned:", len(res))
    for idx,(doc,score) in enumerate(res):
        print(f"\nResult {idx} | score={score} | source={doc.metadata.get('source')}")
        print("Preview:", doc.page_content.replace('\\n',' ')[:500])
except Exception as e:
    print("Query failed:", repr(e))

# Try a very broad query (some text from sample)
try:
    sample_preview = ""
    if os.path.exists("data/sample.txt"):
        with open("data/sample.txt", "r", encoding="utf-8") as fh:
            sample_preview = fh.read().strip()[:80]
    if sample_preview:
        print("\nTrying a broad query using sample text preview:", sample_preview)
        res2 = query_vectorstore(sample_preview, k=10)
        print("Broad query returned:", len(res2))
        for idx,(doc,score) in enumerate(res2):
            print(f"  Broad result {idx} | score={score} | source={doc.metadata.get('source')}")
    else:
        print("No sample preview available for broad query test.")
except Exception as e:
    print("Broad query test failed:", repr(e))

print("\nDiagnostic complete.")