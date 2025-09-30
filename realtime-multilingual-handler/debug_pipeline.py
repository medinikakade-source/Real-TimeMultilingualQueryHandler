# debug_pipeline.py
import traceback
import time
from pathlib import Path

print("DEBUG: starting debug_pipeline.py")
start_ts = time.time()

try:
    from backend.pipeline import load_texts, build_vectorstore, query_vectorstore
    print("Imported pipeline functions OK")
except Exception as e:
    print("ERROR importing backend.pipeline:", e)
    traceback.print_exc()
    raise SystemExit(1)

# 1) show files and sizes
p = Path("data")
files = list(p.glob("*.txt"))
print("Data files found:", [str(x) + f" (size={x.stat().st_size})" for x in files])

# 2) quick chunk check (matching pipeline logic)
try:
    docs = load_texts("data")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    total_chunks = 0
    for d in docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(d["page_content"])
        print(f"File {d['metadata']['source']} -> {len(chunks)} chunk(s)")
        total_chunks += len(chunks)
    print("Total chunks:", total_chunks)
except Exception as e:
    print("Error during chunking check:", e)
    traceback.print_exc()

# 3) build vectordb using a SMALLER test model to speed up and reduce memory.
#    If this runs successfully, we can re-run with the multilingual model.
test_model = "sentence-transformers/all-MiniLM-L6-v2"
print("\nWill run build_vectorstore() with fast test_model:", test_model)
try:
    t0 = time.time()
    # call build_vectorstore with explicit args
    build_vectorstore(data_dir="data", persist_directory="chroma_db_debug", model_name=test_model, collection_name="kb_collection")
    print("build_vectorstore() completed successfully (debug). Time:", time.time() - t0)
except Exception as e:
    print("build_vectorstore() raised an exception:")
    traceback.print_exc()

# 4) run a quick query against the debug DB
try:
    print("\nQuerying debug DB...")
    res = query_vectorstore("How does this handler work?", k=5, persist_directory="chroma_db_debug", model_name=test_model, collection_name="kb_collection")
    print("Query returned count:", len(res))
    for i,(doc,score) in enumerate(res):
        print(f"Result {i} | score={score} | source={doc.metadata.get('source')}")
        print("Preview:", doc.page_content[:300].replace("\n"," "))
except Exception as e:
    print("Query against debug DB failed:")
    traceback.print_exc()

print("\nDEBUG: finished in %.1f seconds" % (time.time() - start_ts))
