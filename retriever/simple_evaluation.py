from pathlib import Path
import yaml
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from retriever.retrievers import load_vector_store_from_config
from retriever.final_retriever import production_retriever, retrieve_FlashrankReranker
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load evaluation set
path = Path(__file__).resolve().parent.parent / "data" / "evaluation_set.yaml"
with open(path, "r") as f:
    evaluation_set = yaml.safe_load(f)


def calculate_recall_at_k(retrieved_ids, relevant_ids, k: int = None) -> float:
    """
    Calculate Recall@K: proportion of relevant documents retrieved.
    """
    if isinstance(relevant_ids, str):
        relevant_ids = [relevant_ids]

    retrieved_set = set(retrieved_ids[:k] if k else retrieved_ids)
    relevant_set = set(relevant_ids)

    if len(relevant_set) == 0:
        return 0.0

    return len(retrieved_set & relevant_set) / len(relevant_set)


def calculate_mrr(retrieved_ids, relevant_ids) -> float:
    """
    Calculate Mean Reciprocal Rank: 1/rank of first relevant document.
    """
    if isinstance(relevant_ids, str):
        relevant_ids = [relevant_ids]

    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0


def evaluate_single_query(retriever, query: str, relevant_ids, k: int = 10, add_query_prefix: bool = False) -> Dict[str, float]:
    """
    Evaluate a single query against a retriever.

    Args:
        retriever: The retriever to evaluate
        query: The search query
        relevant_ids: Ground truth relevant document IDs
        k: Number of documents to retrieve
        add_query_prefix: If True, adds "query: " prefix to the query (for dense/hybrid embeddings)
    """
    # Add "query: " prefix if requested (for dense/hybrid embedding models)
    if add_query_prefix:
        query = f"query: {query}"

    start_time = time.perf_counter()
    documents = retriever.invoke(query)
    end_time = time.perf_counter()

    query_time_ms = (end_time - start_time) * 1000

    retrieved_ids = [doc.metadata.get("_id", "") for doc in documents]

    metrics = {
        f"recall@{k}": calculate_recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": calculate_mrr(retrieved_ids, relevant_ids),
        "query_time_ms": query_time_ms,
        "num_docs_retrieved": len(documents)
    }

    return metrics


def simple_evaluation(
    retriever_configs: List[Tuple[str, str, float, bool, Optional[int], Optional[float], Optional[str], Optional[bool]]],
    request_pool: List[Dict] = None,
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Simple evaluation of multiple retrievers on all queries (no bootstrap).

    Args:
        retriever_configs: List of tuples (retrieval_mode, search_type, threshold, use_rerank, rerank_top_n, rerank_threshold, rerank_model, add_query_prefix)
            - retrieval_mode: "hybrid", "dense", or "sparse"
            - search_type: "similarity" or "similarity_score_threshold"
            - threshold: Score threshold (0 = no threshold). If > 0, uses similarity_score_threshold automatically
            - use_rerank: Whether to apply reranking
            - rerank_top_n: Top N results to keep after reranking
            - rerank_threshold: Minimum rerank score threshold
            - rerank_model (optional): FlashRank model name (e.g., "ms-marco-TinyBERT-L-2-v2")
            - add_query_prefix (optional): If True, adds "query: " prefix for dense/hybrid embeddings
        request_pool: List of dicts with keys 'question', 'answer', 'location'
        k: Number of documents to retrieve

    Returns:
        Dictionary mapping retriever_name to aggregated metrics
    """
    if request_pool is None:
        request_pool = evaluation_set

    results = {}

    for config in retriever_configs:
        # Parse config
        if len(config) == 3:
            retrieval_mode, search_type, threshold = config
            use_rerank, rerank_top_n, rerank_threshold, rerank_model, add_query_prefix = False, None, None, None, False
        elif len(config) == 6:
            retrieval_mode, search_type, threshold, use_rerank, rerank_top_n, rerank_threshold = config
            rerank_model, add_query_prefix = None, False
        elif len(config) == 7:
            retrieval_mode, search_type, threshold, use_rerank, rerank_top_n, rerank_threshold, rerank_model = config
            add_query_prefix = False
        else:
            retrieval_mode, search_type, threshold, use_rerank, rerank_top_n, rerank_threshold, rerank_model, add_query_prefix = config

        if retrieval_mode not in ["hybrid", "dense", "sparse"]:
            print(f"Warning: Invalid retrieval_mode '{retrieval_mode}', skipping")
            continue

        # Create retriever
        if threshold > 0 or search_type == "similarity_score_threshold":
            retriever = production_retriever(
                k=k,
                threshold=threshold if threshold > 0 else None,
                retrieval_mode=retrieval_mode
            )
        else:
            vector_store = load_vector_store_from_config("RAG", force_retrieval_mode=retrieval_mode)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

        # Apply reranking if requested
        if use_rerank:
            if rerank_top_n is None:
                rerank_top_n = k
            if rerank_threshold is None:
                rerank_threshold = 0.5
            if rerank_model is None:
                rerank_model = "ms-marco-MiniLM-L-12-v2"

            retriever = retrieve_FlashrankReranker(
                retriever=retriever,
                model_name=rerank_model,
                top_n=rerank_top_n,
                threshold=rerank_threshold
            )
            # Include model name in retriever_name for clarity
            model_short = rerank_model.replace("ms-marco-", "").replace("-v2", "")
            retriever_name = f"{retrieval_mode}_{search_type}_t{threshold}_rerank_{model_short}_top{rerank_top_n}_t{rerank_threshold}"
            if add_query_prefix:
                retriever_name += "_queryprefix"
        else:
            retriever_name = f"{retrieval_mode}_{search_type}_t{threshold}"
            if add_query_prefix:
                retriever_name += "_queryprefix"

        print(f"\nEvaluating {retriever_name}...")

        metrics_list = []

        for i, query_item in enumerate(request_pool):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(request_pool)} queries")

            query = query_item["question"]
            relevant_ids = query_item["location"]

            metrics = evaluate_single_query(retriever, query, relevant_ids, k, add_query_prefix=add_query_prefix)
            metrics_list.append(metrics)

        # Aggregate metrics
        aggregated = {}
        metric_names = [f"recall@{k}", "mrr", "query_time_ms", "num_docs_retrieved"]

        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            aggregated[f"{metric_name}_mean"] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_min"] = np.min(values)
            aggregated[f"{metric_name}_max"] = np.max(values)
            aggregated[f"{metric_name}_median"] = np.median(values)

        results[retriever_name] = aggregated
        print(f"  ✓ Completed {len(request_pool)} queries")

    return results


def print_simple_evaluation_results(results: Dict[str, Dict[str, float]], k: int):
    """
    Print evaluation results in a readable format.
    """
    print("\n" + "="*80)
    print("RETRIEVER EVALUATION RESULTS (Single Pass)")
    print("="*80)

    for retriever_name, metrics in results.items():
        print(f"\n{retriever_name}:")
        print("-" * 60)

        print(f"  recall@{k:2d}        : {metrics[f'recall@{k}_mean']:.4f} ± {metrics[f'recall@{k}_std']:.4f} "
              f"(min: {metrics[f'recall@{k}_min']:.4f}, max: {metrics[f'recall@{k}_max']:.4f}, median: {metrics[f'recall@{k}_median']:.4f})")

        print(f"  mrr              : {metrics['mrr_mean']:.4f} ± {metrics['mrr_std']:.4f} "
              f"(min: {metrics['mrr_min']:.4f}, max: {metrics['mrr_max']:.4f}, median: {metrics['mrr_median']:.4f})")

        print(f"  query_time_ms    : {metrics['query_time_ms_mean']:.2f}ms ± {metrics['query_time_ms_std']:.2f}ms "
              f"(min: {metrics['query_time_ms_min']:.2f}ms, max: {metrics['query_time_ms_max']:.2f}ms, median: {metrics['query_time_ms_median']:.2f}ms)")

        print(f"  docs_retrieved   : {metrics['num_docs_retrieved_mean']:.1f} ± {metrics['num_docs_retrieved_std']:.1f} "
              f"(min: {int(metrics['num_docs_retrieved_min'])}, max: {int(metrics['num_docs_retrieved_max'])}, median: {metrics['num_docs_retrieved_median']:.1f})")

    print("\n" + "="*80)


if __name__ == "__main__":
    retriever_configs = [

        # Dense/Hybrid WITH "query: " prefix (for embedding models that expect it)
        ("hybrid", "similarity", 0.1, False, None, None, None, True),
        ("dense", "similarity", 0.1, False, None, None, None, True),
        ("sparse", "similarity", 0, False, None, None, None, False),

        # Retrievers with thresholds (non-rerank)
        ("hybrid", "similarity", 0.6, False, None, None, None, True),
        ("dense", "similarity", 0.6, False, None, None, None, True),

        # Reranked retrievers
        ("hybrid", "similarity", 0.6, True, 30, 0, "ms-marco-TinyBERT-L-2-v2", True),
#        ("hybrid", "similarity", 0, True, 10, 0, "ms-marco-MiniLM-L-12-v2", True),
    ]

    k = 20

    print(f"Starting simple evaluation with k={k}")
    print(f"Number of queries: {len(evaluation_set)}")

    results = simple_evaluation(
        retriever_configs=retriever_configs,
        k=k
    )

    print_simple_evaluation_results(results, k)

"""
k=30
================================================================================
RETRIEVER EVALUATION RESULTS (Single Pass)
================================================================================

hybrid_similarity_t0_queryprefix:
------------------------------------------------------------
  recall@30        : 0.9615 ± 0.1923 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5426 ± 0.3876 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 162.23ms ± 43.52ms (min: 87.62ms, max: 280.98ms, median: 153.86ms)
  docs_retrieved   : 30.0 ± 0.0 (min: 30, max: 30, median: 30.0)

dense_similarity_t0_queryprefix:
------------------------------------------------------------
  recall@30        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4202 ± 0.4175 (min: 0.0000, max: 1.0000, median: 0.2000)
  query_time_ms    : 122.51ms ± 43.13ms (min: 77.26ms, max: 286.50ms, median: 107.26ms)
  docs_retrieved   : 30.0 ± 0.0 (min: 30, max: 30, median: 30.0)

sparse_similarity_t0_queryprefix:
------------------------------------------------------------
  recall@30        : 0.9423 ± 0.2332 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5487 ± 0.4027 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 53.85ms ± 21.75ms (min: 32.34ms, max: 122.69ms, median: 45.29ms)
  docs_retrieved   : 30.0 ± 0.0 (min: 30, max: 30, median: 30.0)

hybrid_similarity_t0.8_queryprefix:
------------------------------------------------------------
  recall@30        : 0.4231 ± 0.4940 (min: 0.0000, max: 1.0000, median: 0.0000)
  mrr              : 0.3846 ± 0.4663 (min: 0.0000, max: 1.0000, median: 0.0000)
  query_time_ms    : 175.17ms ± 55.74ms (min: 89.17ms, max: 355.44ms, median: 173.89ms)
  docs_retrieved   : 1.1 ± 0.8 (min: 0, max: 3, median: 1.0)

sparse_similarity_t0.8:
------------------------------------------------------------
  recall@30        : 0.9615 ± 0.1923 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5927 ± 0.3885 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 77.95ms ± 34.30ms (min: 37.28ms, max: 199.44ms, median: 68.01ms)
  docs_retrieved   : 30.0 ± 0.0 (min: 30, max: 30, median: 30.0)

dense_similarity_t0.8_queryprefix:
------------------------------------------------------------
  recall@30        : 0.8269 ± 0.3783 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4281 ± 0.4192 (min: 0.0000, max: 1.0000, median: 0.2250)
  query_time_ms    : 110.30ms ± 34.02ms (min: 72.52ms, max: 218.43ms, median: 102.88ms)
  docs_retrieved   : 19.3 ± 12.0 (min: 0, max: 30, median: 30.0)

hybrid_similarity_t0_rerank_TinyBERT-L-2_top30_t0.7_queryprefix:
------------------------------------------------------------
  recall@30        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5417 ± 0.4152 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 497.87ms ± 87.74ms (min: 281.02ms, max: 775.10ms, median: 490.52ms)
  docs_retrieved   : 12.5 ± 8.8 (min: 0, max: 30, median: 10.5)

hybrid_similarity_t0_rerank_MiniLM-L-12_top30_t0.7_queryprefix:
------------------------------------------------------------
  recall@30        : 0.9038 ± 0.2948 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.6586 ± 0.3789 (min: 0.0000, max: 1.0000, median: 1.0000)
  query_time_ms    : 7298.37ms ± 1202.18ms (min: 2933.96ms, max: 9635.94ms, median: 7060.88ms)
  docs_retrieved   : 16.5 ± 8.7 (min: 1, max: 30, median: 15.0)

================================================================================

k=15 
================================================================================
RETRIEVER EVALUATION RESULTS (Single Pass)
================================================================================

hybrid_similarity_t0_queryprefix:
------------------------------------------------------------
  recall@15        : 0.9038 ± 0.2948 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5354 ± 0.3936 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 151.81ms ± 33.45ms (min: 82.38ms, max: 242.23ms, median: 154.66ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

dense_similarity_t0_queryprefix:
------------------------------------------------------------
  recall@15        : 0.7692 ± 0.4213 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4156 ± 0.4218 (min: 0.0000, max: 1.0000, median: 0.2000)
  query_time_ms    : 102.39ms ± 24.01ms (min: 71.23ms, max: 179.15ms, median: 95.91ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

sparse_similarity_t0:
------------------------------------------------------------
  recall@15        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5172 ± 0.3905 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 68.31ms ± 18.88ms (min: 28.50ms, max: 100.37ms, median: 78.89ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

hybrid_similarity_t0.6_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8462 ± 0.3608 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5765 ± 0.4012 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 160.82ms ± 36.50ms (min: 96.27ms, max: 246.03ms, median: 155.22ms)
  docs_retrieved   : 7.7 ± 1.0 (min: 5, max: 10, median: 8.0)

dense_similarity_t0.8_queryprefix:
------------------------------------------------------------
  recall@15        : 0.7308 ± 0.4436 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4236 ± 0.4235 (min: 0.0000, max: 1.0000, median: 0.2250)
  query_time_ms    : 180.14ms ± 79.41ms (min: 87.24ms, max: 444.12ms, median: 155.66ms)
  docs_retrieved   : 11.2 ± 5.4 (min: 0, max: 15, median: 15.0)

hybrid_similarity_t0_rerank_TinyBERT-L-2_top15_t0.7_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8077 ± 0.3941 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5411 ± 0.4170 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 388.04ms ± 87.67ms (min: 216.73ms, max: 586.78ms, median: 381.03ms)
  docs_retrieved   : 8.1 ± 4.6 (min: 0, max: 15, median: 8.0)

hybrid_similarity_t0_rerank_MiniLM-L-12_top15_t0.7_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8462 ± 0.3608 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.6678 ± 0.4081 (min: 0.0000, max: 1.0000, median: 1.0000)
  query_time_ms    : 3590.18ms ± 925.63ms (min: 1227.71ms, max: 6131.06ms, median: 3798.34ms)
  docs_retrieved   : 10.1 ± 3.9 (min: 1, max: 15, median: 10.0)

================================================================================

================================================================================
RETRIEVER EVALUATION RESULTS (Single Pass)
================================================================================
k=20

hybrid_similarity_t0.1_queryprefix:
------------------------------------------------------------
  recall@15        : 0.9615 ± 0.1923 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5824 ± 0.3790 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 347.60ms ± 119.04ms (min: 184.52ms, max: 683.40ms, median: 333.16ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

dense_similarity_t0.1_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8269 ± 0.3783 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4438 ± 0.4081 (min: 0.0000, max: 1.0000, median: 0.2500)
  query_time_ms    : 260.86ms ± 84.38ms (min: 149.14ms, max: 540.76ms, median: 240.95ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

sparse_similarity_t0.1:
------------------------------------------------------------
  recall@15        : 0.9038 ± 0.2948 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5900 ± 0.3925 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 162.37ms ± 155.78ms (min: 73.68ms, max: 888.96ms, median: 113.82ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

hybrid_similarity_t0.6_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8462 ± 0.3608 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5793 ± 0.4003 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 424.32ms ± 143.59ms (min: 164.36ms, max: 936.57ms, median: 412.88ms)
  docs_retrieved   : 7.7 ± 1.0 (min: 5, max: 10, median: 8.0)

dense_similarity_t0.6_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8269 ± 0.3783 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4438 ± 0.4081 (min: 0.0000, max: 1.0000, median: 0.2500)
  query_time_ms    : 179.40ms ± 84.25ms (min: 70.86ms, max: 580.11ms, median: 168.14ms)
  docs_retrieved   : 15.0 ± 0.0 (min: 15, max: 15, median: 15.0)

hybrid_similarity_t0_rerank_TinyBERT-L-2_top8_t0_queryprefix:
------------------------------------------------------------
  recall@15        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5571 ± 0.4028 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 724.82ms ± 221.07ms (min: 268.82ms, max: 1241.63ms, median: 687.66ms)
  docs_retrieved   : 8.0 ± 0.0 (min: 8, max: 8, median: 8.0)

================================================================================


================================================================================
RETRIEVER EVALUATION RESULTS (Single Pass)
================================================================================

hybrid_similarity_t0.1_queryprefix:
------------------------------------------------------------
  recall@20        : 0.9615 ± 0.1923 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5649 ± 0.3680 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 385.48ms ± 103.13ms (min: 217.84ms, max: 627.85ms, median: 380.26ms)
  docs_retrieved   : 20.0 ± 0.0 (min: 20, max: 20, median: 20.0)

dense_similarity_t0.1_queryprefix:
------------------------------------------------------------
  recall@20        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4461 ± 0.4058 (min: 0.0000, max: 1.0000, median: 0.2500)
  query_time_ms    : 275.42ms ± 57.87ms (min: 156.33ms, max: 511.49ms, median: 270.36ms)
  docs_retrieved   : 20.0 ± 0.0 (min: 20, max: 20, median: 20.0)

sparse_similarity_t0:
------------------------------------------------------------
  recall@20        : 0.8846 ± 0.3195 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5182 ± 0.3893 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 92.25ms ± 39.59ms (min: 34.91ms, max: 268.51ms, median: 79.66ms)
  docs_retrieved   : 20.0 ± 0.0 (min: 20, max: 20, median: 20.0)

hybrid_similarity_t0.6_queryprefix:
------------------------------------------------------------
  recall@20        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.5771 ± 0.3897 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 379.18ms ± 97.47ms (min: 247.46ms, max: 671.08ms, median: 358.12ms)
  docs_retrieved   : 7.8 ± 1.0 (min: 5, max: 10, median: 8.0)

dense_similarity_t0.6_queryprefix:
------------------------------------------------------------
  recall@20        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.4461 ± 0.4058 (min: 0.0000, max: 1.0000, median: 0.2500)
  query_time_ms    : 317.87ms ± 99.99ms (min: 196.46ms, max: 771.68ms, median: 300.17ms)
  docs_retrieved   : 20.0 ± 0.0 (min: 20, max: 20, median: 20.0)

hybrid_similarity_t0.6_rerank_TinyBERT-L-2_top30_t0_queryprefix:
------------------------------------------------------------
  recall@20        : 0.8654 ± 0.3413 (min: 0.0000, max: 1.0000, median: 1.0000)
  mrr              : 0.6124 ± 0.3858 (min: 0.0000, max: 1.0000, median: 0.5000)
  query_time_ms    : 550.37ms ± 202.35ms (min: 261.42ms, max: 1408.36ms, median: 496.55ms)
  docs_retrieved   : 7.8 ± 1.0 (min: 5, max: 10, median: 8.0)

================================================================================
  """
