import json
import numpy as np
from typing import List, Dict

def evaluate_latency(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Calculates latency percentiles for pipeline stages."""
    stages = ["stt", "embedding", "dense", "bm25", "fusion", "rerank", "generation", "validation", "total"]
    metrics = {stage: [] for stage in stages}
    
    for res in results:
        timings = res.get("timings", {})
        for stage in stages:
            if stage in timings:
                metrics[stage].append(timings[stage])
                
    percentiles = {}
    for stage, times in metrics.items():
        if times:
            percentiles[stage] = {
                "p50": np.percentile(times, 50),
                "p70": np.percentile(times, 70),
                "p95": np.percentile(times, 95),
                "p99": np.percentile(times, 99),
                "mean": np.mean(times)
            }
            
    return percentiles
