"""
Utility functions for the RAG system.

This module contains helper functions for document formatting,
context preparation, and response processing.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string.

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted string with all document contents
    """
    if not docs:
        return "No relevant context found."

    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        # Extract metadata
        doc_title = doc.metadata.get('doc_title', 'Unknown Standard')
        title = doc.metadata.get('title', '')
        subtitle = doc.metadata.get('subtitle', '')

        # Create header
        header = f"[Source {i}: {doc_title}"
        if title:
            header += f" - {title}"
        if subtitle:
            header += f" - {subtitle}"
        header += "]"

        # Add content
        formatted_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


def format_docs_simple(docs: List[Document]) -> str:
    """
    Simple document formatting without metadata headers.

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted string with document contents only
    """
    if not docs:
        return "No relevant context found."

    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_scores(docs_with_scores: List[tuple]) -> str:
    """
    Format documents that include relevance scores.

    Args:
        docs_with_scores: List of (document, score) tuples

    Returns:
        Formatted string with documents and their scores
    """
    if not docs_with_scores:
        return "No relevant context found."

    formatted_parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        doc_title = doc.metadata.get('doc_title', 'Unknown Standard')
        title = doc.metadata.get('title', '')

        header = f"[Source {i} - Relevance: {score:.2f}] {doc_title}"
        if title:
            header += f" - {title}"

        formatted_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


def extract_source_info(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract source information from retrieved documents.

    Args:
        docs: List of retrieved documents

    Returns:
        List of dictionaries containing source metadata
    """
    sources = []
    for doc in docs:
        source_info = {
            'standard': doc.metadata.get('doc_title', 'Unknown'),
            'section': doc.metadata.get('title', 'Unknown'),
            'subsection': doc.metadata.get('subtitle', None),
            'chunk_id': doc.metadata.get('chunk_id', None)
        }
        sources.append(source_info)

    return sources


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on their IDs.

    Args:
        docs: List of documents

    Returns:
        List of documents with duplicates removed
    """
    seen_ids = set()
    deduped = []

    for doc in docs:
        doc_id = doc.metadata.get('_id', None)
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            deduped.append(doc)
        elif not doc_id:
            # If no ID, keep the document anyway
            deduped.append(doc)

    return deduped


def create_context_dict(question: str, retriever) -> Dict[str, str]:
    """
    Create a context dictionary for the RAG chain.

    Args:
        question: User's question
        retriever: The retriever to use for fetching documents

    Returns:
        Dictionary with 'context' and 'question' keys
    """
    docs = retriever.invoke(question)
    docs = deduplicate_docs(docs)  # Remove duplicates
    context = format_docs(docs)

    return {
        "context": context,
        "question": question
    }


def prepare_response_with_sources(answer: str, docs: List[Document]) -> Dict[str, Any]:
    """
    Prepare a complete response with answer and source information.

    Args:
        answer: The generated answer
        docs: Retrieved documents used to generate the answer

    Returns:
        Dictionary containing answer and source information
    """
    sources = extract_source_info(docs)

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(docs)
    }
