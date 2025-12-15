import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunking_text(path: str | None = None, chunk_size: int = 800, chunk_overlap: int = 80, include_metadata_in_content: bool = True):
    """
    Load JSON docs from the metadata file, split content into overlapping chunks, and return LangChain documents.

    Args:
        path: Optional path to the metadata JSON file. Defaults to `data/metadatas` in the project root.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
        include_metadata_in_content: If True, prepend title/subtitle to content for better semantic search.
    """

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

            # Create context header
            if context_parts:
                context_header = " > ".join(context_parts) + "\n\n"
                text = context_header + text

        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": elem.get("source"),
                        "type": elem.get("type"),
                        "title": elem.get("title"),
                        "subtitle": elem.get("subtitle"),
                        "subsection": elem.get("subsection"),
                        "subsubsection": elem.get("subsubsection"),
                        "chunk_id": idx,
                    },
                )
            )

    return docs