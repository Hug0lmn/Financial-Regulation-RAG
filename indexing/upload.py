import uuid

def upload_points(vector_store, list_docs : list, batch_size: int = 64) :
    """
    Build PointStructs from chunked_text and provided vectors, then upsert into the collection.
    """

    #Add documents perform automatically the embedding and upload to Qdrant, no need to call specific embedding functions
    #I know that currently the ids are random and doesn't allow easy retrieval and rewriting, this will be fixed later by modifying the chunking function to return also metadata with unique identifiers
    ids = [uuid.uuid4() for text in list_docs]

    for i in range(0, len(ids), batch_size):
        chunk = list_docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        vector_store.add_documents(
            documents=chunk,
            ids=batch_ids,
        )
    
    return