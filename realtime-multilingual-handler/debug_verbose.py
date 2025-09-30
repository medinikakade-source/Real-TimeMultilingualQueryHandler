# debug_verbose.py
import time, sys, traceback
from pathlib import Path

def now():
    return time.strftime("%H:%M:%S")

print(f"{now()} - DEBUG_VERBOSE START", flush=True)

try:
    print(f"{now()} - checking paths...", flush=True)
    print("cwd:", Path.cwd(), flush=True)
    print("data files:", list(Path("data").glob("*.txt")), flush=True)
    print("chroma_db exists:", Path("chroma_db").exists(), "chroma_db_debug exists:", Path("chroma_db_debug").exists(), flush=True)
except Exception as e:
    print("path check error:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

# Import pipeline functions
try:
    print(f"{now()} - importing backend.pipeline", flush=True)
    from backend.pipeline import query_vectorstore, build_vectorstore
    print(f"{now()} - imported functions OK", flush=True)
except Exception as e:
    print(f"{now()} - import ERROR:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

# We'll time each stage explicitly and flush output immediately
try:
    print(f"{now()} - Start: creating embedding object (will be timed)", flush=True)
    t0 = time.time()
    # Create an embedding function once (use a small model to be quick)
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    t1 = time.time()
    print(f"{now()} - Embedding object created in {t1-t0:.2f}s", flush=True)
except Exception as e:
    print(f"{now()} - embedding creation FAILED:", flush=True)
    traceback.print_exc()
    # continue — query_vectorstore will create its own embedding, but we want the traceback
    # sys.exit(1)

# Now try to open the debug Chroma (explicit)
try:
    print(f"{now()} - Attempting to run query_vectorstore against chroma_db_debug", flush=True)
    t0 = time.time()
    res = query_vectorstore(
        "How does the multilingual handler work?",
        k=5,
        persist_directory="chroma_db_debug",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="kb_collection"
    )
    t1 = time.time()
    print(f"{now()} - query_vectorstore returned in {t1-t0:.2f}s", flush=True)
    print("TYPE:", type(res), "LEN:", (len(res) if hasattr(res, '__len__') else "no-len"), flush=True)
    print("REPR (truncated):", repr(res)[:1000], flush=True)
    if res:
        for i,(doc,score) in enumerate(res):
            print(f"--- RESULT {i} ---", flush=True)
            print("score:", score, flush=True)
            print("source:", getattr(doc, "metadata", None), flush=True)
            print("preview:", getattr(doc, "page_content", "")[:400].replace("\n"," "), flush=True)
    else:
        print("Result list empty.", flush=True)
except Exception as e:
    print(f"{now()} - query_vectorstore raised EXCEPTION:", flush=True)
    traceback.print_exc()
    # Attempt to rebuild debug DB now and show progress
    try:
        print(f"{now()} - Attempting rebuild of chroma_db_debug (fast model)", flush=True)
        build_vectorstore(data_dir="data", persist_directory="chroma_db_debug", model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="kb_collection")
        print(f"{now()} - rebuild complete, attempting query again...", flush=True)
        res2 = query_vectorstore("How does the multilingual handler work?", k=5, persist_directory="chroma_db_debug", model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="kb_collection")
        print("After rebuild: type/len:", type(res2), len(res2), flush=True)
        print(repr(res2)[:1000], flush=True)
    except Exception as e2:
        print(f"{now()} - rebuild also failed:", flush=True)
        traceback.print_exc()

print(f"{now()} - DEBUG_VERBOSE END", flush=True)
