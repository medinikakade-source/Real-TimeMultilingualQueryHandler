from backend.pipeline import query_vectorstore
res = query_vectorstore("How does the multilingual handler work?", k=3)
for doc, score in res:
    print("SCORE:", score)
    print("SOURCE:", doc.metadata.get("source"))
    print(doc.page_content[:400])
    print("-" * 40)
