from retriever.retrievers import load_vector_store_from_config

def production_retriever(k=10, threshold=0.6, retrieval_mode = "hybrid") :
    vector_store = load_vector_store_from_config("rag_financial",force_retrieval_mode=retrieval_mode)

    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":k, "score_threshold" : threshold})

    return retriever