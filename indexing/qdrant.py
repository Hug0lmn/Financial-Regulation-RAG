import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

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

def check_collection_type(client, collection_name: str) -> str:
    """
    Check the type of a Qdrant collection (dense, sparse, or both).
    """
    collection = client.get_collection(collection_name)
    has_sparse = collection.config.model_dump()["params"]["sparse_vectors"] is not None

    try : 
        collection.config.model_dump()["params"]["vectors"][""]
        has_dense = True
    except :
        has_dense = False

    if has_dense and has_sparse:
        return "both"
    elif has_dense:
        return "dense"
    elif has_sparse:
        return "sparse"
    else:
        return "none"

def guess_collection_type(model_dense = None, model_sparse = None) :
    
    if model_dense is not None and model_sparse is not None :
        return "both"
    elif model_dense is not None :
        return "dense"
    elif model_sparse is not None :
        return "sparse"
    else :
        raise ValueError("At least one of model_dense or model_sparse must be provided.")

def create_qdrant_collection(client, collection_name : str, model_dense = None, model_sparse = None) :
    """
    Create a Qdrant collection named by collection_name configured for dense, sparse, or both modes.
    """

    try :
        if client.get_collection(collection_name=collection_name) :
            return ValueError(f"A collection already exist with {collection_name}") 
    except :

        mode = guess_collection_type(model_dense, model_sparse) #Guess type

        if mode == "dense" :             
            client.create_collection(collection_name=collection_name,
                vectors_config={"": models.VectorParams(size=model_dense.size,distance=models.Distance.COSINE)
                                })
    
        elif mode == "sparse" :
            client.create_collection(collection_name=collection_name,
                sparse_vectors_config={"langchain-sparse": models.SparseVectorParams()
                                })

        elif mode == "both" :
            client.create_collection(collection_name=collection_name,
                vectors_config={"": models.VectorParams(size=model_dense.size, distance=models.Distance.COSINE)},
                sparse_vectors_config={"langchain-sparse": models.SparseVectorParams()
                                })
        else : 
            raise ValueError(f"Unsupported mode '{mode}'. Use one of: dense, sparse, both.")

def upload_points(list_docs : list, vector_store) :
    """
    Build PointStructs from chunked_text and provided vectors, then upsert into the collection.
    """

    #Add documents perform automatically the embedding and upload to Qdrant, no need to call specific embedding functions
    #I know that currently the ids are random and doesn't allow easy retrieval and rewriting, this will be fixed later by modifying the chunking function to return also metadata with unique identifiers
    ids = [uuid.uuid4()  for text in list_docs]
    vector_store.add_documents(documents=list_docs, ids=ids)    
    
    return (list_docs,ids)

def delete_collection(client, collection_name : str = None) :
    """
    List collections, prompt for indices (hyphen-separated), and delete selected collections.
    If collection_name is None, list all collections and prompt user for deletion.
    """
    
    if collection_name :
        if client.delete_collection(collection_name) : 
            print(f"Deleted successfully : {collection_name}")
        else :
            print(f"Not deleted : {collection_name}")
        return
    
    else :
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
