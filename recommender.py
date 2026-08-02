
import numpy as np

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Computes cosine similarity between two numerical feature vectors.
    Returns a score between 0.0 (unrelated) and 1.0 (identical).
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class RecommenderEngine:
    def __init__(self, metadata_list, mining_engine):
        """
        Initializes Content-Based and Collaborative Filtering engines.
        """
        self.metadata = {m["doc_id"]: m for m in metadata_list}
        self.mining_engine = mining_engine
        self.doc_ids = list(self.metadata.keys())
        self.num_docs = len(self.doc_ids)
        
        # Build dense TF-IDF matrix for similarity calculations
        self.doc_vectors = self._build_dense_vector_matrix()
        
        # Build User Interaction Matrix for Collaborative Filtering simulation
        self.user_interaction_matrix = self._generate_simulated_user_interactions()

    def _build_dense_vector_matrix(self):
        """
        Converts dictionary-based TF-IDF scores into dense NumPy array matrices 
        for fast vector geometry computations.
        """
        vocab = self.mining_engine.vocabulary
        matrix = []
        
        for doc_id in self.doc_ids:
            tfidf_dict = self.mining_engine.tfidf_matrix.get(doc_id, {})
            # Map TF-IDF weights to vocabulary array indices
            vector = [tfidf_dict.get(word, 0.0) for word in vocab]
            matrix.append(vector)
            
        return np.array(matrix)

    def get_content_based_recommendations(self, target_doc_id, top_n=5):
        """
        Finds articles most similar to target_doc_id based on TF-IDF Cosine Similarity.
        """
        if target_doc_id not in self.doc_ids or self.num_docs <= 1:
            return []
            
        target_idx = self.doc_ids.index(target_doc_id)
        target_vector = self.doc_vectors[target_idx]
        
        similarity_scores = []
        for idx, doc_id in enumerate(self.doc_ids):
            if doc_id == target_doc_id:
                continue  # Skip self-recommendation
                
            sim_score = calculate_cosine_similarity(target_vector, self.doc_vectors[idx])
            similarity_scores.append((doc_id, float(sim_score)))
            
        # Sort by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for doc_id, score in similarity_scores[:top_n]:
            recommendations.append({
                "doc_id": doc_id,
                "title": self.metadata[doc_id]["title"],
                "source": self.metadata[doc_id]["source"],
                "similarity_score": score,
                "category": self.mining_engine.profile_and_classify_document(doc_id)
            })
            
        return recommendations

    def _generate_simulated_user_interactions(self, num_users=20):
        """
        Simulates a User-Item Interaction Matrix representing user clicks/reads 
        to enable Collaborative Filtering testing.
        """
        np.random.seed(42)  # Fixed seed for repeatable academic testing
        # Matrix shape: (num_users, num_documents) with values 0 (not read) or 1 (read)
        interaction_matrix = np.random.choice([0, 1], size=(num_users, self.num_docs), p=[0.8, 0.2])
        return interaction_matrix

    def get_collaborative_recommendations(self, target_user_id=0, top_n=5):
        """
        User-Based Collaborative Filtering: finds similar users based on reading history
        and recommends articles read by those similar users.
        """
        if target_user_id >= len(self.user_interaction_matrix):
            return []

        user_vector = self.user_interaction_matrix[target_user_id]
        user_similarities = []

        # Find user similarity across all other simulated users
        for other_user_id, other_vector in enumerate(self.user_interaction_matrix):
            if other_user_id == target_user_id:
                continue
            sim = calculate_cosine_similarity(user_vector, other_vector)
            user_similarities.append((other_user_id, sim))

        # Sort to find top most similar users
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = user_similarities[:5]

        # Predict article scores based on similar users' reads
        predicted_item_scores = np.zeros(self.num_docs)
        for other_user_id, sim_weight in top_similar_users:
            predicted_item_scores += sim_weight * self.user_interaction_matrix[other_user_id]

        # Filter out items the target user has already read
        recommendations = []
        for idx, score in enumerate(predicted_item_scores):
            doc_id = self.doc_ids[idx]
            if user_vector[idx] == 0 and score > 0:  # Unread by target user
                recommendations.append((doc_id, score))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, cf_score in recommendations[:top_n]:
            results.append({
                "doc_id": doc_id,
                "title": self.metadata[doc_id]["title"],
                "source": self.metadata[doc_id]["source"],
                "cf_score": float(cf_score),
                "category": self.mining_engine.profile_and_classify_document(doc_id)
            })

        return results

    def get_hybrid_recommendations(self, target_doc_id, target_user_id=0, alpha=0.6, top_n=5):
        """
        Combines Content-Based (alpha weight) and Collaborative Filtering ((1-alpha) weight) 
        into a unified recommendation rank list.
        """
        cb_recs = {r["doc_id"]: r["similarity_score"] for r in self.get_content_based_recommendations(target_doc_id, top_n=self.num_docs)}
        cf_recs = {r["doc_id"]: r["cf_score"] for r in self.get_collaborative_recommendations(target_user_id, top_n=self.num_docs)}

        # Normalize CF scores to [0, 1] range for score fusion
        max_cf = max(cf_recs.values()) if cf_recs and max(cf_recs.values()) > 0 else 1.0
        normalized_cf = {doc_id: score / max_cf for doc_id, score in cf_recs.items()}

        hybrid_scores = []
        for doc_id in self.doc_ids:
            if doc_id == target_doc_id:
                continue
            
            cb_score = cb_recs.get(doc_id, 0.0)
            cf_score = normalized_cf.get(doc_id, 0.0)
            
            # Hybrid Fusion Formula
            fused_score = (alpha * cb_score) + ((1.0 - alpha) * cf_score)
            
            hybrid_scores.append({
                "doc_id": doc_id,
                "title": self.metadata[doc_id]["title"],
                "source": self.metadata[doc_id]["source"],
                "hybrid_score": fused_score,
                "cb_score": cb_score,
                "cf_score": cf_score,
                "category": self.mining_engine.profile_and_classify_document(doc_id)
            })

        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_scores[:top_n]

# --- Terminal Testing Block ---
if __name__ == "__main__":
    from crawler import run_crawler_pipeline
    from text_mining import TextMiningEngine

    print("🚀 Testing Recommender System Module...\n")
    metadata, contents, _ = run_crawler_pipeline(use_fallback=True)
    mining_engine = TextMiningEngine(contents)

    recommender = RecommenderEngine(metadata, mining_engine)

    print("📌 CONTENT-BASED RECOMMENDATIONS (Articles similar to Doc_1):")
    cb_results = recommender.get_content_based_recommendations("Doc_1", top_n=3)
    for res in cb_results:
        print(f"  • [{res['doc_id']}] {res['title'][:50]}... | Cosine Sim: {res['similarity_score']:.4f}")

    print("\n👥 COLLABORATIVE FILTERING RECOMMENDATIONS (For User 0):")
    cf_results = recommender.get_collaborative_recommendations(target_user_id=0, top_n=3)
    for res in cf_results:
        print(f"  • [{res['doc_id']}] {res['title'][:50]}... | CF Score: {res['cf_score']:.4f}")

    print("\n⚡ HYBRID RECOMMENDATIONS (Fused Content + Collaborative):")
    hybrid_results = recommender.get_hybrid_recommendations("Doc_1", target_user_id=0, alpha=0.6, top_n=3)
    for res in hybrid_results:
        print(f"  • [{res['doc_id']}] {res['title'][:50]}... | Hybrid Score: {res['hybrid_score']:.4f}")