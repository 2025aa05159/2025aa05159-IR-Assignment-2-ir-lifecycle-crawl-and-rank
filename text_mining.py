import re
import math
import time
from collections import Counter

# --- NLTK Imports & Automatic Resource Handling ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- Scikit-Learn Imports for Comparative Analysis (Task C) ---
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Quietly check and fetch required NLTK resources for Streamlit Cloud compatibility
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        if 'punkt' in resource:
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
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

# ==============================================================================
# CLASS 1: TEXT MINING ENGINE (Used for primary indexing and keyword search)
# ==============================================================================
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
        and overall TF-IDF weights across the article collection.
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
        ranked_keywords = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_keywords[:top_n]

    def profile_and_classify_document(self, doc_id):
        """
        Classifies an article into a tech category by comparing token frequencies 
        against domain keyword buckets.
        """
        words = self.parsed_tokens.get(doc_id, [])
        
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

# ==============================================================================
# CLASS 2: TEXT MINER (Used for Task C comparative benchmarking)
# ==============================================================================
class TextMiner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text, normalization_strategy='stemming'):
        """Cleans and normalizes text based on Stemming or Lemmatization."""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]

        if normalization_strategy == 'stemming':
            tokens = [self.stemmer.stem(word) for word in tokens]
        elif normalization_strategy == 'lemmatization':
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)

    def build_features(self, documents, norm_strategy='stemming', vectorizer_type='tfidf'):
        """Builds the feature matrix and returns performance metrics for Task C comparison."""
        start_time = time.time()
        
        processed_docs = [self.preprocess_text(doc, norm_strategy) for doc in documents]
        
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer()
        else:
            raise ValueError("Unsupported vectorizer type. Choose 'tfidf' or 'count'.")
            
        feature_matrix = vectorizer.fit_transform(processed_docs)
        vocab_size = len(vectorizer.get_feature_names_out())
        end_time = time.time()
        
        return {
            'matrix': feature_matrix,
            'vectorizer': vectorizer,
            'vocab_size': vocab_size,
            'time_taken': end_time - start_time,
            'processed_docs': processed_docs
        }
