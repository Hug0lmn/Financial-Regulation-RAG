import joblib
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List
from langchain_core.embeddings import Embeddings

#from sentence_transformers import SentenceTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer

def output_supported_models(type : str):
    """
    Small helper function to output supported models from fastembed
    """

    if type == "dense" :
        return [[i["model"],f"Size:{i['size_in_GB']}", f'Dim {i["dim"]}', f'Desc {i["description"]}'] for i in TextEmbedding.list_supported_models()]
    elif type == "sparse" :
        return [[i["model"],f"Size:{i['size_in_GB']}", f'Desc {i["description"]}'] for i in SparseTextEmbedding.list_supported_models()]


class FastEmbedEmbeddings(Embeddings):
    """Lightweight wrapper around `fastembed` dense embeddings for LangChain.

    Provides document and query embedding helpers backed by `TextEmbedding`.

    Args:
        model_name: FastEmbed model id to load (defaults to BGE small English v1.5).

    Attributes:
        model: Underlying `TextEmbedding` instance.
        size: Dimensionality of the generated embeddings.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.size = self.model.get_embedding_size(model_name=model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(embedding) for embedding in self.model.embed(texts)]
    
    def embed_query(self, text: str) -> List[float]:
        return list(next(self.model.embed([text])))




#As of right now the two functions below are not used anymore since the embedding and upload to Qdrant is done directly by the vector_store
#There also no more needs to store the vectorizer as consequence

def get_dense_vectors(chunked_data : list) :
    """
    Encode chunked_data content with SentenceTransformer and return list of dense vector lists.
    """

    dense_model = SentenceTransformer('all-MiniLM-L6-v2')

    dense_vectors = []
    for x in chunked_data :
        dense_vectors.append(dense_model.encode(x["content"]).tolist())

    return dense_vectors

def get_sparse_vectors(chunked_text : list, vectorizer_name : str = "") :
    """
    Fit TF-IDF on chunked_text content, save vectorizer next to this file, return tfidf_matrix.
    """

    all_contents = [doc["content"] for doc in chunked_text]

    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(all_contents)

    out_dir = Path(__file__).resolve().parent
    filename = f"vectorizer_{vectorizer_name}.joblib" if vectorizer_name else "vectorizer.joblib"
    joblib.dump(vectorizer, out_dir / filename)

    return tfidf_matrix
