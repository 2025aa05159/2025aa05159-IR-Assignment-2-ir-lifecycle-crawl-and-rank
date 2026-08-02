
"""
Information Retrieval Engine - Task F: Evaluation Metrics
=========================================================
This module computes standard IR performance metrics to evaluate 
the effectiveness of our search and recommendation ranking algorithms.
"""

import numpy as np
import math

class IREvaluator:
    def __init__(self):
        """
        Initializes the evaluation engine.
        """
        pass

    def calculate_precision_recall_f1(self, retrieved_docs, relevant_docs):
        """
        Calculates standard Precision, Recall, and F1-Score for a set of retrieved documents.
        """
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        # Precision: proportion of retrieved documents that are relevant
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        
        # Recall: proportion of relevant documents that were successfully retrieved
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        
        # F1-Score: Harmonic mean of precision and recall
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return precision, recall, f1_score

    def calculate_metrics_at_k(self, retrieved_docs, relevant_docs, k=5):
        """
        Calculates Precision@K and Recall@K (evaluating only the top K results).
        """
        top_k_retrieved = retrieved_docs[:k]
        return self.calculate_precision_recall_f1(top_k_retrieved, relevant_docs)[:2]

    def calculate_average_precision(self, retrieved_docs, relevant_docs):
        """
        Calculates Average Precision (AP) for a single query.
        Used to compute MAP (Mean Average Precision) across multiple queries.
        """
        relevant_set = set(relevant_docs)
        if not relevant_set:
            return 0.0

        running_correct_count = 0
        precision_sum = 0.0

        for rank_idx, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                running_correct_count += 1
                current_precision = running_correct_count / (rank_idx + 1)
                precision_sum += current_precision

        return precision_sum / len(relevant_set)

    def calculate_mrr(self, retrieved_docs, relevant_docs):
        """
        Calculates the Reciprocal Rank (RR) for a single query.
        RR = 1 / rank of the FIRST relevant document found.
        """
        relevant_set = set(relevant_docs)
        
        for rank_idx, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                return 1.0 / (rank_idx + 1)
                
        return 0.0

    def calculate_ndcg(self, retrieved_docs, relevant_docs, k=5):
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG@K).
        Measures ranking quality by penalizing relevant documents that appear lower in the list.
        """
        top_k_retrieved = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(top_k_retrieved):
            if doc_id in relevant_set:
                relevance_score = 1.0  # Binary relevance for this assignment
                dcg += relevance_score / math.log2(i + 2) # i+2 because rank is 1-indexed (i+1) + 1 in formula
                
        # Calculate IDCG (Ideal DCG - if all relevant docs were ranked perfectly at the top)
        idcg = 0.0
        ideal_hits = min(len(relevant_set), k)
        for i in range(ideal_hits):
            idcg += 1.0 / math.log2(i + 2)
            
        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg

    def generate_full_evaluation_report(self, query_results, ground_truth):
        """
        Takes a list of retrieved documents and a ground truth list, 
        returning a dictionary of all required Task F metrics.
        """
        p, r, f1 = self.calculate_precision_recall_f1(query_results, ground_truth)
        p_at_5, r_at_5 = self.calculate_metrics_at_k(query_results, ground_truth, k=5)
        ap = self.calculate_average_precision(query_results, ground_truth)
        rr = self.calculate_mrr(query_results, ground_truth)
        ndcg = self.calculate_ndcg(query_results, ground_truth, k=5)

        return {
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1_Score": round(f1, 4),
            "Precision@5": round(p_at_5, 4),
            "Recall@5": round(r_at_5, 4),
            "Average_Precision (MAP base)": round(ap, 4),
            "MRR": round(rr, 4),
            "NDCG@5": round(ndcg, 4)
        }

# --- Terminal Testing Block ---
if __name__ == "__main__":
    print("🚀 Testing Task F: Evaluation Metrics Engine...\n")
    
    evaluator = IREvaluator()
    
    # 1. Simulate a search query result
    simulated_retrieved_docs = ["Doc_1", "Doc_10", "Doc_3", "Doc_45", "Doc_12", "Doc_8"]
    
    # 2. Simulate Ground Truth (what the ideal result should have been)
    simulated_ground_truth = ["Doc_3", "Doc_12", "Doc_1"]
    
    # 3. Generate Report
    report = evaluator.generate_full_evaluation_report(simulated_retrieved_docs, simulated_ground_truth)
    
    print("📊 EVALUATION METRICS REPORT (Simulated Query):")
    for metric_name, value in report.items():
        print(f"  • {metric_name}: {value}")