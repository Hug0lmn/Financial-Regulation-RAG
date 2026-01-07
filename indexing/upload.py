from tqdm import tqdm
from langchain_core.documents import Document
from pathlib import Path
import json

def transfo_list_into_Document(list_chunk, use_prefix: bool = False, prefix: str = "passage: ") :
    """
    Transform list of chunks into LangChain Document objects.

    Args:
        list_chunk: List of dictionaries containing document data
        use_prefix: Whether to add a prefix to the page_content (default: False)
        prefix: The prefix to add if use_prefix=True (default: "passage: ")

    Returns:
        Tuple of (docs, list_ids)
    """
    docs = []
    list_ids = []
    for elem in list_chunk :
        # Get content and optionally add prefix
        content = elem.get("content")
        if use_prefix and content:
            content = prefix + content

        doc = Document(page_content=content,
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

def upload_points(vector_store, batch_size: int = 50, use_prefix: bool = False, prefix: str = "passage: ") :
    """
    Upload documents to the vector store.

    Args:
        vector_store: The vector store to upload to
        batch_size: Number of documents to upload per batch (default: 50)
        use_prefix: Whether to add a prefix to document content (default: False)
                   Set to True for models like snowflake-arctic-embed that require prefixes
        prefix: The prefix to add to documents (default: "passage: ")

    Example:
        # Without prefix (for models like bge-base-en-v1.5)
        upload_points(vector_store, batch_size=50, use_prefix=False)

        # With prefix (for models like snowflake-arctic-embed-m)
        upload_points(vector_store, batch_size=50, use_prefix=True, prefix="passage: ")
    """
    file = Path(__file__).resolve().parent.parent / "data" / "metadatas"

    with open(file, "r", encoding="utf-8") as f:
        list_docs = json.load(f)

    docs, ids = transfo_list_into_Document(list_docs, use_prefix=use_prefix, prefix=prefix)

    for i in tqdm(range(0, len(ids), batch_size), desc="Uploading batches"):
        chunk = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        vector_store.add_documents(
            documents=chunk,
            ids=batch_ids,
        )

    return