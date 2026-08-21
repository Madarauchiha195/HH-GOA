import json
from typing import List, Dict

def evaluate_grounding(results: List[Dict]) -> Dict[str, float]:
    """Evaluates grounding metrics."""
    metrics = {
        "grounding_rate": 0.0,
        "abstention_rate": 0.0,
    }
    
    total = len(results)
    if total == 0:
        return metrics
        
    grounded_count = 0
    abstention_count = 0
    
    for res in results:
        generation = res.get("generation", {})
        if not generation.get("should_answer", True):
            abstention_count += 1
        elif generation.get("grounded", False):
            grounded_count += 1
            
    metrics["grounding_rate"] = grounded_count / max(total - abstention_count, 1)
    metrics["abstention_rate"] = abstention_count / total
    
    return metrics
