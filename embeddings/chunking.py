import json
import joblib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def chunking_text(path : str, chunk_size = 800, chunk_overlap = 80) :
    """Load JSON docs from path, split each content into overlapping chunks, return chunk metadata list."""

    with open(path) as f :
        meta = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ". ", "; "]
    )

    chunked_metadata = []

    for elem in meta :
        text = elem["content"]
        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            chunked_metadata.append({
                "doc_title" : elem.get("doc_title"),
                "title" : elem.get('title'),
                "subtitle" : elem.get("subtitle"),
                "subsection" : elem.get("subsection"),
                "subsubsection" : elem.get("subsubsection"),
                "content" : chunk,
                "chunk_id": idx 
            })

    return chunked_metadata

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
