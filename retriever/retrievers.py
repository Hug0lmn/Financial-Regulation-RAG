"""
Retriever utilities for loading and configuring vector stores.

This module provides functions to load QdrantVectorStore instances
from configuration files with different retrieval modes.
"""

import yaml
from pathlib import Path
from typing import Optional, Tuple
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

from embeddings.embedding import FastEmbedEmbeddings
from indexing.qdrant import load_qdrant_client

path = Path(__file__).parent.parent
good_path = path/ "indexing/collections.yaml"


def load_vector_store_from_config(
    collection_name: str,
    client: Optional[QdrantClient] = None,
    config_path: str = str(good_path),
    force_retrieval_mode: Optional[str] = None
) -> QdrantVectorStore:
    """
    Load a QdrantVectorStore from configuration file.

    Args:
        collection_name: Name of the collection to load
        client: Optional QdrantClient. If None, will be loaded automatically
        config_path: Path to the collections config YAML file
        force_retrieval_mode: Force a specific retrieval mode ("dense", "sparse", or "hybrid").
                             If None, mode is determined from available embeddings in config.
                             Useful to test same collection with different retrieval strategies.

    Returns:
        Configured QdrantVectorStore

    Raises:
        ValueError: If collection not found, no embeddings configured, or invalid mode
        FileNotFoundError: If config file doesn't exist
    """
    # Load client if not provided
    if client is None:
        client = load_qdrant_client()

    # Load config
    config_file_path = Path(config_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Find collection config
    model_config = None
    for model in config.get("models", []):
        if model.get("name") == collection_name:
            model_config = model
            break

    if model_config is None:
        available = [m["name"] for m in config.get("models", [])]
        raise ValueError(
            f"Collection '{collection_name}' not found in config. "
            f"Available collections: {', '.join(available)}"
        )

    # Load embedding models based on force_retrieval_mode or config
    model_dense = None
    model_sparse = None

    # Determine which models to load
    if force_retrieval_mode:
        force_mode = force_retrieval_mode.lower()
        if force_mode not in ["dense", "sparse", "hybrid"]:
            raise ValueError(
                f"Invalid force_retrieval_mode: '{force_retrieval_mode}'. "
                "Must be 'dense', 'sparse', or 'hybrid'."
            )

        # Load only required models based on forced mode
        if force_mode in ["dense", "hybrid"]:
            if model_config.get("dense") is not None:
                dense_name = model_config["dense"]["name"]
                model_dense = FastEmbedEmbeddings(model_name=dense_name)
            elif force_mode == "dense":
                raise ValueError(f"Dense embeddings not configured for '{collection_name}'")

        if force_mode in ["sparse", "hybrid"]:
            if model_config.get("sparse") is not None:
                sparse_name = model_config["sparse"]["name"]
                model_sparse = FastEmbedSparse(model_name=sparse_name)
            elif force_mode == "sparse":
                raise ValueError(f"Sparse embeddings not configured for '{collection_name}'")
    else:
        # Load all available models from config
        if model_config.get("dense") is not None:
            dense_name = model_config["dense"]["name"]
            model_dense = FastEmbedEmbeddings(model_name=dense_name)

        if model_config.get("sparse") is not None:
            sparse_name = model_config["sparse"]["name"]
            model_sparse = FastEmbedSparse(model_name=sparse_name)

    # Determine retrieval mode
    if force_retrieval_mode:
        # Use forced mode
        mode_map = {
            "dense": RetrievalMode.DENSE,
            "sparse": RetrievalMode.SPARSE,
            "hybrid": RetrievalMode.HYBRID
        }
        retrieval_mode = mode_map[force_retrieval_mode.lower()]
    else:
        # Auto-detect from loaded models
        if model_dense and model_sparse:
            retrieval_mode = RetrievalMode.HYBRID
        elif model_dense:
            retrieval_mode = RetrievalMode.DENSE
        elif model_sparse:
            retrieval_mode = RetrievalMode.SPARSE
        else:
            raise ValueError(f"No embedding models configured for collection '{collection_name}'")

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        retrieval_mode=retrieval_mode,
        embedding=model_dense,
        sparse_embedding=model_sparse
    )

    mode_suffix = f" (forced: {force_retrieval_mode})" if force_retrieval_mode else ""
    print(f"✓ Vector store loaded: {collection_name} (mode: {retrieval_mode.value}{mode_suffix})")

    return vector_store