import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, ndcg_score
from typing import Dict, List

class RecommendationEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def precision_at_k(self, recommended: List, relevant: List, k: int = 10) -> float:
        """Calculate Precision@K"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        if len(recommended_k) == 0:
            return 0.0
        
        return len(relevant_set & recommended_set) / len(recommended_k)
    
    def recall_at_k(self, recommended: List, relevant: List, k: int = 10) -> float:
        """Calculate Recall@K"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        recommended_set = set(recommended_k)
        
        if len(relevant_set) == 0:
            return 0.0
        
        return len(relevant_set & recommended_set) / len(relevant_set)
    
    def ndcg_at_k(self, recommended: List, relevance_scores: Dict, k: int = 10) -> float:
        """Calculate NDCG@K"""
        recommended_k = recommended[:k]
        scores = [relevance_scores.get(item, 0) for item in recommended_k]
        
        if sum(scores) == 0:
            return 0.0
        
        # DCG
        dcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
        
        # IDCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(idx + 2) for idx, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_recommendations(self, test_data: pd.DataFrame, 
                               model, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Comprehensive evaluation of recommendation model"""
        results = {f"precision@{k}": [] for k in k_values}
        results.update({f"recall@{k}": [] for k in k_values}
        results.update({f"ndcg@{k}": [] for k in k_values}
        
        for user_id in test_data['user_id'].unique():
            user_data = test_data[test_data['user_id'] == user_id]
            relevant_items = user_data[user_data['rating'] >= 4]['activity_id'].tolist()
            
            if len(relevant_items) == 0:
                continue
                
            recommendations = model.predict(user_id, n_recommendations=max(k_values))
            relevance_scores = dict(zip(user_data['activity_id'], user_data['rating']))
            
            for k in k_values:
                precision = self.precision_at_k(recommendations, relevant_items, k)
                recall = self.recall_at_k(recommendations, relevant_items, k)
                ndcg = self.ndcg_at_k(recommendations, relevance_scores, k)
                
                results[f"precision@{k}"].append(precision)
                results[f"recall@{k}"].append(recall)
                results[f"ndcg@{k}"].append(ndcg)
        
        # Calculate averages
        final_results = {}
        for metric, values in results.items():
            final_results[metric] = np.mean(values) if values else 0.0
        
        return final_results