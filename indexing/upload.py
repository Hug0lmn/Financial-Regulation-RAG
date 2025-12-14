import uuid

# Custom namespace UUID for this project (you can change this to any UUID you want)
CHUNK_NAMESPACE = uuid.UUID('12945478-1334-5278-1234-567292345678')

def create_chunk_id(metadata: dict, max_len: int = 4) -> uuid.UUID:
    """
    Create a unique, reconstructible UUID from metadata fields.
    Uses UUID v5 (SHA-1 hash) to generate deterministic UUIDs from metadata.

    Args:
        metadata: Dictionary containing chunk metadata
        max_len: Maximum number of characters to take from each field

    Returns:
        UUID that can be reconstructed from the same metadata
    """
    def truncate(value, length=max_len):
        """Get first 'length' chars if exists, else '0'"""
        if value and str(value).strip():
            return str(value)[:length].replace("__", "_")
        return "0"

    def get_name(value) :
        if value and str(value).strip():
            return "".join([name for name in str(value) if name.isupper() or name.isdigit()])
        return "0"

    parts = [
        metadata.get("source"),
        metadata.get("type"),
        truncate(metadata.get("title")),
        truncate(metadata.get("subtitle")),
        truncate(metadata.get("subsection")),
        truncate(metadata.get("subsubsection")),
        str(metadata.get("chunk_id", 0))
    ]

    # Create a deterministic string ID
    string_id = "_".join(parts)

    # Convert to UUID v5 (deterministic, based on SHA-1 hash)
    return uuid.uuid5(CHUNK_NAMESPACE, string_id)


def upload_points(vector_store, list_docs : list, batch_size: int = 100) :
    """
    Build PointStructs from chunked_text and provided vectors, then upsert into the collection.
    Uses metadata-based IDs for easy reconstruction and retrieval.
    """

    # Create IDs from metadata instead of random UUIDs
    ids = [create_chunk_id(doc.metadata) for doc in list_docs]

    for i in range(0, len(ids), batch_size):
        print(f"{i} / {len(ids)}")
        chunk = list_docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        vector_store.add_documents(
            documents=chunk,
            ids=batch_ids,
        )
    
    return