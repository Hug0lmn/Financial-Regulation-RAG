import os
from typing import Literal

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

def load_qdrant_client() :
    """
    Load and return a QdrantClient using QDRANT_API_KEY and QDRANT_URL from .env.
    """
    
    load_dotenv() 

    api_key = os.getenv("QDRANT_API_KEY")
    url = os.getenv("QDRANT_URL")

    qdrant_client = QdrantClient(
        url= url, 
        api_key= api_key,
    )
    
    return qdrant_client

def create_qdrant_collection(client, collection_name : str, mode : Literal["dense","sparse","both"]) :
    """
    Create a Qdrant collection named by collection_name configured for dense, sparse, or both modes.
    """

    try : 
        if client.get_collection(collection_name=collection_name) :
            raise ValueError(f"A collection already exist with {collection_name}") 
    except :

        if mode == "dense" : 
            client.create_collection(collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=384,distance=models.Distance.COSINE)
                                })
    
        elif mode == "sparse" :
            client.create_collection(collection_name=collection_name,
                sparse_vectors_config={"text": models.SparseVectorParams()
                                })

        elif mode == "both" :
            client.create_collection(collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
                sparse_vectors_config={"text": models.SparseVectorParams()
                                })
        else : 
            raise ValueError(f"Unsupported mode '{mode}'. Use one of: dense, sparse, both.")

def upload_points(chunked_text : list, client, collection_name : str, dense_vectors = None, sparse_vectors = None) :
    """
    Build PointStructs from chunked_text and provided vectors, then upsert into the collection.
    """

    points = []

    for idx in range(len(chunked_text)):
        
        elem = chunked_text[idx]
        vector_dict = {}
        
        if dense_vectors is not None:
            vector_dict["dense"] = dense_vectors[idx]
        
        if sparse_vectors is not None:
            vectorizer, tfidf_matrix = sparse_vectors
            sparse_vector = tfidf_matrix[idx]
            vector_dict["text"] = models.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.data.tolist()
            )

        point = PointStruct(
            id=idx,
            vector=vector_dict,
            payload=elem
        )
    
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Sucessfully uploaded data to Qdrant with {collection_name}")

    return 

def delete_collection(client) :
    """List collections, prompt for indices (hyphen-separated), and delete selected collections."""
    
    all_col = client.get_collections()
    list_col = [elem.name for elem in all_col.collections]

    print(f"List col : {list_col}")
    col_nb = input(fr"What collection would you want to delete (Type exact position, for multiple deletion separate by -)")

    cols = col_nb.split("-")
    
    if len(cols) <= len(list_col) :
        for col in cols :
            col = int(col)
            
            #Send back False if not deleted
            if client.delete_collection(list_col[col]) : 
                print(f"Deleted successfully : {list_col[col]}")
            else :
                print(f"Not deleted : {list_col[col]}")
    
    else : 
        print("Listed more than available nb of collection")
