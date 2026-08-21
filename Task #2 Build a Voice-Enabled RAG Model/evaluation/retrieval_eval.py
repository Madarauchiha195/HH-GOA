import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def evaluate_retrieval(results: List[Dict]) -> Dict[str, float]:
    """Evaluates retrieval metrics (Recall@K, MRR@K)."""
    metrics = {
        "recall_1": 0.0,
        "recall_5": 0.0,
        "recall_10": 0.0,
        "mrr_10": 0.0
    }
    
    total = len(results)
    if total == 0:
        return metrics
        
    for res in results:
        expected_docs = set(res.get("expected_docs", []))
        retrieved_docs = [doc.get("document_id") for doc in res.get("retrieved", [])]
        
        # Recall @ K
        for k in [1, 5, 10]:
            top_k = retrieved_docs[:k]
            if any(doc in expected_docs for doc in top_k):
                metrics[f"recall_{k}"] += 1
                
        # MRR @ 10
        for rank, doc in enumerate(retrieved_docs[:10]):
            if doc in expected_docs:
                metrics["mrr_10"] += 1.0 / (rank + 1)
                break
                
    # Average
    for k in metrics:
        metrics[k] /= total
        
    return metrics

if __name__ == "__main__":
    # Test script for eval
    print(evaluate_retrieval([
        {"expected_docs": ["doc1"], "retrieved": [{"document_id": "doc1"}, {"document_id": "doc2"}]}
    ]))
