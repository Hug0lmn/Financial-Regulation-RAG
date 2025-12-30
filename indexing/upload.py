from tqdm import tqdm
from langchain_core.documents import Document

def transfo_list_into_Document(list_chunk) :
    
    docs = []
    list_ids = []
    for elem in list_chunk :
        doc = Document(page_content=elem.get("content"),
                       metadata = {
                        "source": elem.get("source"),
                        "type": elem.get("type"),
                        "title": elem.get("title"),
                        "subtitle": elem.get("subtitle"),
                        "subsection": elem.get("subsection"),
                        "subsubsection": elem.get("subsubsection"),
                        "chunk_id" : elem.get("chunk_id")
                       }) 

        docs.append(doc)
        list_ids.append(elem.get("qdrant_id"))
    
    return docs, list_ids

def upload_points(vector_store, list_docs : list, batch_size: int = 50) :

    docs, ids = transfo_list_into_Document(list_docs)

    for i in tqdm(range(0, len(ids), batch_size), desc="Uploading batches"):
        chunk = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        vector_store.add_documents(
            documents=chunk,
            ids=batch_ids,
        )
    
    return