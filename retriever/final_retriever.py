from retriever.retrievers import load_vector_store_from_config
from qdrant_client import models
from flashrank import Ranker, RerankRequest

filters = models.Filter(must=[models.FieldCondition(key="metadata.type", match=models.MatchValue(value="main"))])

def production_retriever(k=10, threshold=None, retrieval_mode = "hybrid", filter=filters) :
    
    if not threshold :
        if retrieval_mode == "hybrid" :
            threshold = 0.6
        elif retrieval_mode == "sparse" : #Sparse tend to have lower similarity score
            threshold = 0.4
        elif retrieval_mode == "dense" :
            threshold = 0.7
    
    vector_store = load_vector_store_from_config("RAG",force_retrieval_mode=retrieval_mode)
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":k, "score_threshold" : threshold,"filter":filter})

    return retriever

class retrieve_FlashrankReranker:
    def __init__(self, retriever, model_name="ms-marco-MiniLM-L-12-v2", top_n=10):

        self.retriever = retriever
        self.ranker = Ranker(model_name=model_name)
        self.top_n = top_n

    def invoke(self, query):
        """
        Retrieve and rerank in one call.
        """

        documents = self.retriever.invoke(query)

        if not documents:
            return []

        return self.rerank(query, documents)

    def rerank(self, query, documents):
        """
        Rerank a list of documents.

        Args:
            query: Search query
            documents: List of documents to rerank
        """

        if not documents:
            return []

        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)[:self.top_n]

        reranked_docs = []
        for result in results:
            doc = documents[result['id']]
            doc.metadata['rerank_score'] = result['score']
            reranked_docs.append(doc)

        return reranked_docs