
import re
import math
from collections import Counter

# --- NLTK Imports & Automatic Resource Handling ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Quietly check and fetch required NLTK resources for Virtual Lab compatibility
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Load standard NLTK English stop words set
NLTK_STOP_WORDS = set(stopwords.words('english'))

def process_and_tokenize_text(text_content):
    """
    Cleans raw document text using NLTK word_tokenize.
    Converts to lowercase, strips non-alphabet characters, and drops standard stop words.
    """
    # Remove HTML tags and special non-letter symbols
    clean_str = re.sub(r'<[^>]+>', ' ', text_content)
    clean_str = re.sub(r'[^a-zA-Z\s]', ' ', clean_str)
    
    # NLTK Tokenization
    raw_tokens = word_tokenize(clean_str.lower())
    
    # Filter out short noise words and NLTK stop words
    filtered_tokens = [
        token for token in raw_tokens 
        if len(token) > 2 and token not in NLTK_STOP_WORDS
    ]
    return filtered_tokens

class TextMiningEngine:
    def __init__(self, document_contents):
        """
        Initializes the mining engine with the crawled document dictionary.
        Format: {"Doc_1": "raw article text...", "Doc_2": "..."}
        """
        self.contents = document_contents
        
        # Tokenize every document body using NLTK
        self.parsed_tokens = {
            doc_id: process_and_tokenize_text(body) 
            for doc_id, body in document_contents.items()
        }
        
        # Build the global vocabulary list across all documents
        self.vocabulary = sorted(list(set(
            word for token_list in self.parsed_tokens.values() for word in token_list
        )))
        
        # Pre-allocate matrices for TF, IDF, and TF-IDF
        self.term_freq_matrix = {}
        self.inverse_doc_freq = {}
        self.tfidf_matrix = {}
        
        # Calculate full corpus TF-IDF representation
        self._calculate_corpus_tfidf()

    def _calculate_corpus_tfidf(self):
        """
        Calculates term frequency (TF), inverse document frequency (IDF), 
        and overall TF-IDF weights across our article collection.
        """
        total_docs = len(self.contents)
        if total_docs == 0:
            return

        doc_word_counts = {}
        docs_containing_word = Counter()

        # Step 1: Calculate Term Frequency (TF) for each document
        for doc_id, words in self.parsed_tokens.items():
            counts = Counter(words)
            doc_word_counts[doc_id] = counts
            doc_len = max(len(words), 1)  # Avoid division by zero
            
            # TF = (Count of word in doc) / (Total words in doc)
            self.term_freq_matrix[doc_id] = {
                word: count / doc_len for word, count in counts.items()
            }
            
            # Track how many distinct documents contain each word
            for unique_word in set(words):
                docs_containing_word[unique_word] += 1

        # Step 2: Calculate Inverse Document Frequency (IDF) globally
        # Formula: IDF(w) = log((Total Docs + 1) / (Docs with w + 1)) + 1
        for word in self.vocabulary:
            matching_docs = docs_containing_word.get(word, 0)
            self.inverse_doc_freq[word] = math.log((total_docs + 1) / (matching_docs + 1)) + 1.0

        # Step 3: Compute final TF-IDF scores (TF * IDF)
        for doc_id in self.contents.keys():
            self.tfidf_matrix[doc_id] = {}
            for word, tf_val in self.term_freq_matrix[doc_id].items():
                self.tfidf_matrix[doc_id][word] = tf_val * self.inverse_doc_freq[word]

    def extract_top_keywords(self, doc_id, top_n=5):
        """
        Pulls the top N keywords with the highest TF-IDF weight for a specific article.
        """
        if doc_id not in self.tfidf_matrix:
            return []
        
        word_scores = self.tfidf_matrix[doc_id]
        # Sort words in descending order of their TF-IDF weight
        ranked_keywords = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_keywords[:top_n]

    def profile_and_classify_document(self, doc_id):
        """
        Classifies an article into a tech category by comparing token frequencies 
        against domain keyword buckets.
        """
        words = self.parsed_tokens.get(doc_id, [])
        
        # Category keyword sets for classification matching
        ai_terms = {'ai', 'neural', 'transformer', 'model', 'openai', 'chatgpt', 'agent', 'intelligence'}
        cloud_terms = {'cloud', 'quantum', 'infrastructure', 'api', 'server', 'data', 'chip', 'hardware'}
        software_terms = {'app', 'android', 'apple', 'google', 'code', 'web', 'browser', 'plugin', 'wordpress'}
        business_terms = {'startup', 'vc', 'venture', 'funding', 'market', 'affiliate', 'monetize', 'invest'}

        cat_scores = {
            'Artificial Intelligence': sum(1 for w in words if w in ai_terms),
            'Cloud & Hardware': sum(1 for w in words if w in cloud_terms),
            'Software & Mobile': sum(1 for w in words if w in software_terms),
            'Business & VC': sum(1 for w in words if w in business_terms)
        }
        
        # Pick category with the most keyword matches
        chosen_category = max(cat_scores, key=cat_scores.get)
        if cat_scores[chosen_category] == 0:
            chosen_category = 'General Tech'
            
        return chosen_category

    def get_corpus_statistics(self):
        """
        Generates corpus-level metrics to render charts and tables in Streamlit.
        """
        all_words = [token for words in self.parsed_tokens.values() for token in words]
        
        word_counts = Counter(all_words).most_common(15)
        
        category_distribution = Counter()
        for doc_id in self.contents.keys():
            assigned_cat = self.profile_and_classify_document(doc_id)
            category_distribution[assigned_cat] += 1
            
        return {
            "total_words_processed": len(all_words),
            "vocabulary_size": len(self.vocabulary),
            "top_frequent_words": word_counts,
            "category_distribution": dict(category_distribution)
        }

# --- Terminal Testing Block ---
if __name__ == "__main__":
    from crawler import run_crawler_pipeline
    
    print("🚀 Testing Refactored NLTK Text Mining Engine...\n")
    _, raw_contents, _ = run_crawler_pipeline(use_fallback=True)
    
    mining_engine = TextMiningEngine(raw_contents)
    corpus_stats = mining_engine.get_corpus_statistics()
    
    print("📊 CORPUS METRICS (NLTK Processed):")
    print(f"  • Total Words Processed: {corpus_stats['total_words_processed']}")
    print(f"  • Vocabulary Size: {corpus_stats['vocabulary_size']}")
    print(f"  • Category Distribution: {corpus_stats['category_distribution']}")
    
    print("\n🏷️ TOP NLTK TF-IDF KEYWORDS FOR Doc_1:")
    top_words = mining_engine.extract_top_keywords("Doc_1", top_n=5)
    for word, tfidf_val in top_words:
        print(f"  • {word}: {tfidf_val:.4f}")