"""
RAG (Retrieval Augmented Generation) module.

This module contains the RAG chain components including:
- Prompts for different use cases
- RAG chain construction
- Utilities for document formatting
"""

from rag.chain import create_rag_chain, get_rag_response

__all__ = ["create_rag_chain", "get_rag_response"]
