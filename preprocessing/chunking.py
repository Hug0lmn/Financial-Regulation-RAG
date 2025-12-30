import json
from pathlib import Path
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom namespace UUID for deterministic chunk IDs
CHUNK_NAMESPACE = uuid.UUID('12945478-1334-5278-1234-567292345678')

def truncate(value, length):
    """Return the first 'length' characters if value is truthy, otherwise '0'."""
    if value and str(value).strip():
        return str(value)[:length].replace("__", "_")
    return "0"

def create_chunk_id(metadata, idx) :
    """
    Build a deterministic UUID5 for a chunk based on its metadata and index.
    """

    parts = [
        metadata.get("source"),
        metadata.get("type"),
        truncate(metadata.get("title"),4),
        truncate(metadata.get("subtitle"),4),
        truncate(metadata.get("subsection"),4),
        truncate(metadata.get("subsubsection"),4),
        truncate(metadata.get("content"),4),
        str(idx)
    ]

    # Create a deterministic string ID
    string_id = "_".join(parts)

    # Convert to UUID v5 (deterministic, based on SHA-1 hash)
    return uuid.uuid5(CHUNK_NAMESPACE, string_id)

def chunking_text(path: str | list | None = None, chunk_size: int = 800, chunk_overlap: int = 80, include_metadata_in_content: bool = True):
    """
    Load metadata entries, optionally prepend the heading context, split into overlapping text chunks,
    and return a list of chunk metadata.

    Args:
        path: Optional path to the metadata JSON file. Defaults to `data/metadatas` in the project root.
        chunk_size: Target number of characters per chunk, depends on the embedding model capabilities.
        chunk_overlap: Number of characters shared between consecutive chunks.
        include_metadata_in_content: If True, prepend title/subtitle to content for better semantic search.
    """

    if type(path) == list :
        meta = path 
    else : 
        meta_dir = path if path else Path(__file__).resolve().parent.parent / "data" / "metadatas"
        with open(meta_dir) as f:
            meta = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"],
    )

    docs = []

    for elem in meta:
        text = elem["content"]

        # Prepend metadata context to content for better embeddings
        if include_metadata_in_content:
            context_parts = []

            # Add title hierarchy
            if elem.get("title"):
                context_parts.append(elem["title"])
            if elem.get("subtitle"):
                context_parts.append(elem["subtitle"])
            if elem.get("subsection"):
                context_parts.append(elem["subsection"])
            if elem.get("subsubsection"):
                context_parts.append(elem["subsubsection"])

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            id_ = create_chunk_id(elem, idx)
            
            # Create context header
            if context_parts:
                context_header = " > ".join(context_parts) + "\n"
                text = context_header + chunk
            
            metadata = {
                "source": elem.get("source"),
                "type": elem.get("type"),
                "title": elem.get("title"),
                "subtitle": elem.get("subtitle"),
                "subsection": elem.get("subsection"),
                "subsubsection": elem.get("subsubsection"),
                "qdrant_id" : str(id_),
                "chunk_id": idx,
                "content": text
                }
            
            docs.append(metadata)

    return docs
