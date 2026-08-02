
"""
Information Retrieval Engine - Task D: Web Searching & Link Analysis
====================================================================
This module handles inverted indexing, web link graph construction, 
PageRank calculation via Power Iteration, HITS (Hubs & Authorities), 
and score-fused query retrieval.
"""

import re
import numpy as np
from collections import defaultdict


class WebSearchEngine:
    def __init__(self, document_metadata_list, document_contents_map, text_mining_engine):
        """
        Initializes the Search Engine with document collections, metadata, 
        and pre-processed tokens from the Text Mining Engine.
        """
        # Map document IDs to their metadata records (title, source URL, etc.)
        self.metadata_lookup = {doc["doc_id"]: doc for doc in document_metadata_list}
        self.document_contents = document_contents_map
        self.mining_engine = text_mining_engine
        
        self.doc_ids = list(document_contents_map.keys())
        self.total_documents = len(self.doc_ids)
        
        # 1. Inverted Index for fast candidate retrieval
        self.inverted_index = defaultdict(set)
        self._construct_inverted_index()
        
        # 2. Web Graph Representation for Link Analysis algorithms
        self.adjacency_matrix = self._build_link_graph()
        
        # 3. Graph Ranking Algorithms (PageRank & HITS)
        self.pagerank_scores = self.compute_pagerank()
        self.hub_scores, self.authority_scores = self.compute_hits_scores()

    def _construct_inverted_index(self):
        """
        Builds an Inverted Index mapping distinct vocabulary tokens to 
        the set of Document IDs where they appear.
        """
        for doc_id, token_list in self.mining_engine.parsed_tokens.items():
            for token in token_list:
                self.inverted_index[token].add(doc_id)

    def _build_link_graph(self):
        """
        Constructs a Web Link Graph (Adjacency Matrix L where L[i][j] = 1 
        if Document i references or links to Document j).
        
        Uses domain cross-referencing and keyword overlap heuristics to simulate 
        hyperlink structures across tech news articles.
        """
        matrix = np.zeros((self.total_documents, self.total_documents))
        
        for source_index, source_doc_id in enumerate(self.doc_ids):
            source_text = self.document_contents[source_doc_id].lower()
            
            for target_index, target_doc_id in enumerate(self.doc_ids):
                if source_index == target_index:
                    continue  # Ignore self-referencing links
                
                # Check for shared key concepts/references between articles
                target_title = self.metadata_lookup[target_doc_id]["title"].lower()
                title_keywords = [word for word in target_title.split() if len(word) > 4]
                
                # If key concepts match, model it as a directed link
                if any(keyword in source_text for keyword in title_keywords[:2]):
                    matrix[source_index][target_index] = 1.0
                    
        # Ensure minimal web connectivity to prevent complete graph disconnection
        if np.sum(matrix) == 0:
            for source_index in range(self.total_documents):
                next_neighbor = (source_index + 1) % self.total_documents
                matrix[source_index][next_neighbor] = 1.0
                
        return matrix

    def compute_pagerank(self, damping_factor=0.85, max_iterations=50, convergence_tolerance=1e-6):
        """
        Calculates PageRank authority scores using the Power Iteration Method.
        
        Formula:
            PR(p) = (1 - d)/N + d * sum(PR(i) / OutDegree(i))
        Where d = damping factor (0.85), N = total documents.
        """
        num_docs = self.total_documents
        if num_docs == 0:
            return {}

        # Initialize stationary probabilities uniformly: 1 / N
        ranks = np.ones(num_docs) / num_docs
        
        # Calculate outgoing degree (number of outgoing links) for each page
        out_degrees = self.adjacency_matrix.sum(axis=1)
        out_degrees[out_degrees == 0] = 1.0  # Avoid division by zero for sink nodes
        
        # Transition Probability Matrix
        transition_matrix = self.adjacency_matrix / out_degrees[:, np.newaxis]

        # Power Iteration Loop until numerical convergence
        for iteration in range(max_iterations):
            previous_ranks = np.copy(ranks)
            
            # Update PageRank vector
            ranks = (1.0 - damping_factor) / num_docs + damping_factor * (transition_matrix.T @ previous_ranks)
            
            # Check L1 norm convergence criteria
            if np.linalg.norm(ranks - previous_ranks, ord=1) < convergence_tolerance:
                break
                
        return {self.doc_ids[i]: float(ranks[i]) for i in range(num_docs)}

    def compute_hits_scores(self, max_iterations=30):
        """
        Calculates HITS (Hyperlink-Induced Topic Search) Hub and Authority scores.
        
        - Authority Score: Quality of content within the page (A = L^T * H).
        - Hub Score: Quality of links pointing to other authority pages (H = L * A).
        """
        num_docs = self.total_documents
        if num_docs == 0:
            return {}, {}

        # Initialize Hubs and Authorities vectors to all ones
        hubs = np.ones(num_docs)
        authorities = np.ones(num_docs)
        
        for _ in range(max_iterations):
            # Step 1: Update Authorities based on incoming Hub links
            authorities = self.adjacency_matrix.T @ hubs
            
            # Step 2: Update Hubs based on outgoing Authority links
            hubs = self.adjacency_matrix @ authorities
            
            # Step 3: Normalize vectors to unit length
            auth_norm = np.linalg.norm(authorities)
            hub_norm = np.linalg.norm(hubs)
            
            if auth_norm > 0:
                authorities /= auth_norm
            if hub_norm > 0:
                hubs /= hub_norm

        hub_dictionary = {self.doc_ids[i]: float(hubs[i]) for i in range(num_docs)}
        authority_dictionary = {self.doc_ids[i]: float(authorities[i]) for i in range(num_docs)}
        
        return hub_dictionary, authority_dictionary

    def execute_search(self, user_query, apply_pagerank_boost=True, top_k_results=10):
        """
        Processes a search query, performs inverted index filtering, and calculates 
        a fused score combining Content Relevance (TF-IDF) and Graph Authority (PageRank).
        """
        # Tokenize and normalize search query terms
        query_terms = re.findall(r'\b[a-zA-Z]{3,}\b', user_query.lower())
        if not query_terms:
            return []

        # Step 1: Retrieve candidate documents via Inverted Index lookup
        matching_document_ids = set()
        for term in query_terms:
            if term in self.inverted_index:
                matching_document_ids.update(self.inverted_index[term])
                
        if not matching_document_ids:
            return []

        search_results = []
        for doc_id in matching_document_ids:
            # Step 2: Compute TF-IDF Content Match Score for search terms
            doc_tfidf_weights = self.mining_engine.tfidf_matrix.get(doc_id, {})
            content_match_score = sum(doc_tfidf_weights.get(term, 0.0) for term in query_terms)
            
            # Step 3: Retrieve PageRank Authority Score
            pr_authority_score = self.pagerank_scores.get(doc_id, 0.0)
            
            # Step 4: Fusion Ranking (0.7 Content Relevance + 0.3 Graph Authority)
            if apply_pagerank_boost:
                final_rank_score = (0.7 * content_match_score) + (0.3 * pr_authority_score * 10.0)
            else:
                final_rank_score = content_match_score
                
            search_results.append({
                "doc_id": doc_id,
                "title": self.metadata_lookup[doc_id]["title"],
                "source": self.metadata_lookup[doc_id]["source"],
                "content_score": content_match_score,
                "pagerank_score": pr_authority_score,
                "final_rank_score": final_rank_score,
                "snippet": self.document_contents[doc_id][:180] + "..."
            })

        # Step 5: Order documents by final fused ranking score descending
        search_results.sort(key=lambda item: item["final_rank_score"], reverse=True)
        return search_results[:top_k_results]


# --- Terminal Verification & Testing Block ---
if __name__ == "__main__":
    from crawler import run_crawler_pipeline
    from text_mining import TextMiningEngine
    
    print("🚀 Testing Task D: Web Search & PageRank Engine...\n")
    metadata, contents, _ = run_crawler_pipeline(use_fallback=True)
    mining_engine = TextMiningEngine(contents)
    
    search_engine = WebSearchEngine(metadata, contents, mining_engine)
    
    print("🕸️ TOP 3 PAGERANK AUTHORITATIVE DOCUMENTS:")
    top_ranked = sorted(search_engine.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for doc_id, pr_val in top_ranked:
        doc_title = search_engine.metadata_lookup[doc_id]['title']
        print(f"  • [{doc_id}] {doc_title[:55]}... | PR Score: {pr_val:.4f}")
        
    print("\n🔍 EXECUTE TEST SEARCH QUERY: 'ai model'")
    results = search_engine.execute_search("ai model", apply_pagerank_boost=True, top_k_results=3)
    for idx, hit in enumerate(results, 1):
        print(f"  {idx}. [{hit['doc_id']}] {hit['title']} (Fused Score: {hit['final_rank_score']:.4f})")