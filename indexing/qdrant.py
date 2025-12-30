import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from indexing.collections_config import store_info_collections, del_collection_yaml

def load_qdrant_client() -> QdrantClient:
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
        if client.get_collection(collection_name) :
            raise ValueError(f"A collection already exists with {collection_name}") 
        #I know that there is a parameter in create_collection to overwrite if existing, but I prefer to be explicit here

    except :
        mode = guess_collection_type(model_dense, model_sparse) #Guess type
        collections_config = store_info_collections(collection_name, model_dense, model_sparse)

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
        
        return collections_config

def delete_collection(client, collection_name : str = None) :
    """
    List collections, prompt for indices (hyphen-separated), and delete selected collections.
    If collection_name is None, list all collections and prompt user for deletion.
    """

    result_qdrant = False
    result_yaml = False

    if collection_name :

        result_qdrant = client.delete_collection(collection_name) #True if deleted else False
        
        if result_qdrant : #Delete yaml only if collection deleted in qdrant
            result_yaml = del_collection_yaml(collection_name)

        if result_yaml : #Implicit result_qdrant = True
            print(f"Deleted successfully on qdrant and yaml: {collection_name}")
        elif result_qdrant and not result_yaml: 
            print(f"Deleted successfully on qdrant but yaml failed: {collection_name}")
        else :
            print(f"Not deleted : {collection_name}")
    
    else :
        all_col = client.get_collections()
        list_col = [elem.name for elem in all_col.collections]

        print(f"List col : {list_col}")
        col_nb = input(fr"What collection would you want to delete (Type exact position, for multiple deletion separate by -)")

        if col_nb : #If answer not empty
            cols = col_nb.split("-")
    
            if len(cols) <= len(list_col) :
                for col in cols :
                    
                    try :
                        col = int(col)
                    except : 
                        raise TypeError("col value not convertible to int")
            
                    #Send back False if not deleted
                    if client.delete_collection(list_col[col]) :

                        if del_collection_yaml(list_col[col]) :
                            print(f"Deleted successfully (qdrant & yaml): {list_col[col]}")
                        else :
                            print(f"Deleted successfully (qdrant): {list_col[col]}")

                    else :
                        print(f"Not deleted : {list_col[col]}")
    
            else : 
                print("Listed more than available nb of collection")