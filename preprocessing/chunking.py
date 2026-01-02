import json
from pathlib import Path
import uuid

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom namespace UUID for deterministic chunk IDs
# Ideally the "seed" will be stocked in .env 
CHUNK_NAMESPACE = uuid.UUID('12945478-1334-5278-1234-567292345678')

def truncate(value, length):
    if value and str(value).strip():
        return str(value)[:length].replace("__", "_")
    return "0"

def create_chunk_id(metadata, idx) :
    #Build a deterministic UUID5 for a chunk based on its metadata.

    parts = [
        metadata.get("source"),
        metadata.get("type"),
        truncate(metadata.get("title"),10),
        truncate(metadata.get("subtitle"),10),
        truncate(metadata.get("subsection"),10),
        truncate(metadata.get("subsubsection"),10),
        truncate(metadata.get("content"),20),
        str(idx)
    ]
    string_id = "_".join(parts)

    return uuid.uuid5(CHUNK_NAMESPACE, string_id)

def chunking_text(
    path: str | list | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
    include_metadata_in_content: bool = True
):
    """
    Load metadata entries, split into token-based overlapping text chunks, and return a list of chunk metadata.

    Args:
        path: Optional path to the metadata JSON file. Defaults to `data/metadatas` in the project root.
        chunk_size: Target number of tokens per chunk (use <512 for BGE-base).
        chunk_overlap: Number of tokens shared between consecutive chunks (overlap happen if, even with the character split, there are chunk that exceed the length limit).
        include_metadata_in_content: If True, keep a lightweight context header in content).
    """

    if type(path) == list :
        meta = path 
    else : 
        meta_dir = path if path else Path(__file__).resolve().parent.parent / "data" / "metadatas"
        with open(meta_dir) as f:
            meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=False,
    )

    docs = []

    for elem in meta:
        text = elem["content"]

        # Prepend metadata context to content for better embeddings
        if include_metadata_in_content:
            context_parts = []

            if elem.get("source") : context_parts.append(elem["source"])
            if elem.get("title"): context_parts.append(elem["title"])
            if elem.get("subtitle"): context_parts.append(elem["subtitle"])
            if elem.get("subsection"): context_parts.append(elem["subsection"])
            if elem.get("subsubsection"): context_parts.append(elem["subsubsection"])

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            id_ = create_chunk_id(elem, idx)
            
            # Create context header
            context_header = None
            if context_parts:
                #Choose the last one + the source
                if len(context_parts) >= 2 :
                    context_header = context_parts[0] + " | " + context_parts[-1] + " | "
                else :
                    context_header = " | ".join(context_parts) + " | "
            
            metadata = {
                "source": elem.get("source"),
                "type": elem.get("type"),
                "title": elem.get("title"),
                "subtitle": elem.get("subtitle"),
                "subsection": elem.get("subsection"),
                "subsubsection": elem.get("subsubsection"),
                "qdrant_id" : str(id_),
                "chunk_id": idx,
                "content": context_header + chunk
                }
            
            docs.append(metadata)

    return docs
