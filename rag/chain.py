"""
RAG Chain implementation.

This module contains the main RAG chain construction and execution logic.
"""

from typing import Optional, Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from prompts import get_prompt_template
from utils import format_docs, deduplicate_docs, prepare_response_with_sources
import sys
from pathlib import Path

# Add parent directory to path to import retriever module
sys.path.insert(0, str(Path(__file__).parent.parent))
from retriever.final_retriever import production_retriever


def create_rag_chain(
    llm,
    retriever=None,
    prompt_type: str = "default",
    k: int = 6,
    threshold: float = 0.6,
    include_sources: bool = False
):
    """
    Create a complete RAG chain.

    Args:
        llm: Language model to use for generation
        retriever: Optional custom retriever. If None, uses production retriever
        prompt_type: Type of prompt template ("default", "detailed", "comparison", "definition")
        k: Number of documents to retrieve (if using default retriever)
        threshold: Similarity threshold for retrieval (if using default retriever)
        include_sources: Whether to include source information in the response

    Returns:
        Configured RAG chain ready for invocation. The chain always returns
        the generated answer along with the formatted context, raw retrieved
        documents, and the exact prompt input passed to the LLM. When
        include_sources=True, source metadata and counts are also included.

    """
    # Get retriever
    if retriever is None:
        retriever = production_retriever(k=k, threshold=threshold)

    # Get prompt template
    prompt = get_prompt_template(prompt_type)

    def retrieve_and_format(question):
        docs = deduplicate_docs(retriever.invoke(question))
        return {
            "context": format_docs(docs),
            "question": question,
            "_docs": docs,
        }

    def model_chain():
        return (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        )

    def base_parallel():
        """
        Parallel branch that keeps the intermediate values we care about while the
        answer is generated. This lets the caller inspect context, docs, and prompt input.
        """
        return RunnableParallel(
            answer=model_chain(),
            context=lambda x: x["context"],
            question=lambda x: x["question"],
            _docs=lambda x: x["_docs"],
            prompt_input=lambda x: {"context": x["context"], "question": x["question"]},
        )

    def process_output_with_sources(output):
        answer = output["answer"]
        docs = output.get("_docs", [])
        response = prepare_response_with_sources(answer, docs)
        response.update(
            {
                "context": output.get("context"),
                "question": output.get("question"),
                "retrieved_documents": docs,
                "prompt_input": output.get("prompt_input"),
            }
        )
        return response

    def process_output_without_sources(output):
        return {
            "answer": output["answer"],
            "context": output.get("context"),
            "question": output.get("question"),
            "retrieved_documents": output.get("_docs", []),
            "prompt_input": output.get("prompt_input"),
        }

    # Create the chain based on whether we need sources
    if include_sources:
        chain = retrieve_and_format | base_parallel() | process_output_with_sources
    else:
        chain = retrieve_and_format | base_parallel() | process_output_without_sources

    return chain