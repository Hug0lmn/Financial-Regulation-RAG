from pathlib import Path
import yaml

def explore_collections_yaml() :
    """
    Load and print the contents of collections.yaml for debugging.
    """
    path = Path(__file__).resolve().parent / "collections.yaml"

    if not path.exists():
        print(f"No collections.yaml found")
        return

    with open(path) as f:
        collection_config = yaml.safe_load(f)
        return collection_config
    
def del_collection_yaml(collection_name) :

    path = Path(__file__).resolve().parent / "collections.yaml"

    #load yaml file
    with open(path) as f :
        collection_config = yaml.safe_load(f)

    #Check each iteration of models
    for idx, col in enumerate(collection_config) : 
        if col["name"] == collection_name :
            collection_config.pop(idx)

            #Save back 
            with open(path, "w") as f :
                yaml.safe_dump(collection_config, f, sort_keys=False)
            return True

    return False
    
def store_info_collections(collection_name : str, model_dense = None, model_sparse = None) :

    """
    Store the collection information in a collections.yaml file, if it doesn't exist, it creates it, 
    otherwise it appends the new collection to the existing file.
    """

    if model_dense :
        dense_dim = model_dense.model.model.model_description.dim
        dense_name = model_dense.model.model.model_name
    
    if model_sparse :
        sparse_name = model_sparse._model.model.model_name
    
    COLLECTIONS_CONFIG = {
            "name" : collection_name, 
            "dense": {
                "name": dense_name,
                "size": dense_dim
            },
            "sparse": {
                "name": sparse_name
            }
        }
    
    path = Path(__file__).resolve().parent
    for file in path.iterdir() : #Check if a collections.yaml file already exists

        if "collections.yaml" in str(file) :

            with open(file) as f:
                collection_config = yaml.safe_load(f)

            if collection_config : #Not empty
                collection_config.append(COLLECTIONS_CONFIG) #No chance that the collection name already exists, so we can append it directly
            else :
                collection_config = []
                collection_config.append(COLLECTIONS_CONFIG)
            
            with open(file, "w") as f:
                yaml.safe_dump(collection_config, f, sort_keys=False)
                print("Collection config updated with new model")
            return
            

    collection_config = {"models": [COLLECTIONS_CONFIG]}
    path_file = path / "collections.yaml"
    with open(path_file, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
        print(f"Collection config saved to {path_file}")
        return