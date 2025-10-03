# debug_query2.py
import traceback
from backend.pipeline import query_vectorstore, build_vectorstore

def run():
    try:
        print("Calling query_vectorstore(...) with debug args")
        res = query_vectorstore(
            "How does the multilingual handler work?",
            k=5,
            persist_directory="chroma_db_debug",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            collection_name="kb_collection"
        )
        print("TYPE OF result:", type(res))
        try:
            print("LEN:", len(res))
        except Exception as e:
            print("Could not get len():", repr(e))
        print("REPR of result:", repr(res))
        # Inspect items if any
        if res:
            for i, item in enumerate(res):
                print(f"--- ITEM {i} repr: {repr(item)}")
                try:
                    doc, score = item
                    print("  item type:", type(item))
                    print("  doc type:", type(doc))
                    print("  has page_content:", hasattr(doc, "page_content"))
                    print("  metadata repr:", repr(getattr(doc, "metadata", None)))
                    print("  score repr/type:", repr(score), type(score))
                    print("  preview:", getattr(doc, "page_content", "")[:500])
                except Exception as e:
                    print("  failed to unpack item:", repr(e))
        else:
            print("Result is empty or falsy.")
    except Exception as exc:
        print("query_vectorstore raised exception:")
        traceback.print_exc()
        print("Attempting to rebuild vectorstore (debug DB)...")
        try:
            build_vectorstore(data_dir="data", persist_directory="chroma_db_debug", model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="kb_collection")
            print("Rebuild attempted — now re-run the query once more (no further rebuilds).")
            r2 = query_vectorstore("How does the multilingual handler work?", k=5, persist_directory="chroma_db_debug", model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="kb_collection")
            print("After rebuild, len:", len(r2))
            print("Repr:", repr(r2))
        except Exception as e2:
            print("Rebuild failed:")
            traceback.print_exc()

if __name__ == "__main__":
    run()
