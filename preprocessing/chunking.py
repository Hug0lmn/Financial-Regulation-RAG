import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunking_text(path: str | None = None, chunk_size: int = 800, chunk_overlap: int = 80):
    """
    Load JSON docs from the metadata file, split content into overlapping chunks, and return LangChain documents.

    Args:
        path: Optional path to the metadata JSON file. Defaults to `data/metadatas` in the project root.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
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
        chunks = splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_title": elem.get("doc_title"),
                        "title": elem.get("title"),
                        "subtitle": elem.get("subtitle"),
                        "subsection": elem.get("subsection"),
                        "subsubsection": elem.get("subsubsection"),
                        "chunk_id": idx,
                    },
                )
            )

    return docs