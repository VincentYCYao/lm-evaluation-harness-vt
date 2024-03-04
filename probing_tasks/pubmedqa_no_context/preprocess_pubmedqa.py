def doc_to_text(doc) -> str:
    # ctxs = "\n".join(doc["CONTEXTS"])
    return "Question: {}\nAnswer:".format(doc["QUESTION"])
    
