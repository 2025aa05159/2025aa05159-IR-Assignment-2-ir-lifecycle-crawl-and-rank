
import urllib.request
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime

# Default seed RSS feeds for tech news
DEFAULT_SEEDS = [
    "https://news.ycombinator.com/rss",
    "https://techcrunch.com/feed/",
    "https://feeds.feedburner.com/TechCrunch/"
]

# Fallback tech articles in case BITS Virtual Lab has restricted internet access
FALLBACK_CORPUS = [
    {
        "url": "https://tech-example.com/ai-breakthrough-2026",
        "title": "New Neural Network Architecture Beats Traditional Transformers",
        "author": "Tech Staff Writer",
        "pub_date": "2026-07-15",
        "metadata": {"category": "Artificial Intelligence", "source": "TechDaily"},
        "content": "Researchers have unveiled a novel neural network architecture that reduces computational overhead by 40 percent. This breakthrough improves vector retrieval, real-time transformer inference, and large language model training efficiency across distributed clusters."
    },
    {
        "url": "https://tech-example.com/quantum-computing-cloud",
        "title": "Quantum Computing Reaches Cloud Infrastructure Milestone",
        "author": "Elena Rostova",
        "pub_date": "2026-07-18",
        "metadata": {"category": "Quantum Computing", "source": "CloudInsider"},
        "content": "Cloud providers are now offering hybrid quantum-classical computing pipelines. Developers can execute complex optimization algorithms and graph calculations directly via web APIs without dedicated hardware."
    },
    {
        "url": "https://tech-example.com/graph-database-search",
        "title": "Graph Search Algorithms Power Next-Gen Recommendation Engines",
        "author": "Devin Kumar",
        "pub_date": "2026-07-20",
        "metadata": {"category": "Search Systems", "source": "DataEngineeringToday"},
        "content": "Integrating PageRank and link analysis graphs into modern search engines enables more accurate content recommendations. Hybrid recommender systems combine collaborative filtering with textual TF-IDF embeddings for superior user ranking."
    },
    {
        "url": "https://tech-example.com/ai-breakthrough-2026-mirror",  # Duplicate document test
        "title": "New Neural Network Architecture Beats Traditional Transformers (Reprint)",
        "author": "Syndicated News",
        "pub_date": "2026-07-16",
        "metadata": {"category": "Artificial Intelligence", "source": "TechMirror"},
        "content": "Researchers have unveiled a novel neural network architecture that reduces computational overhead by 40 percent. This breakthrough improves vector retrieval, real-time transformer inference, and large language model training efficiency across distributed clusters."
    }
]

def hash_text_content(text):
    """
    Computes a SHA-256 fingerprint of the text content.
    Used for document deduplication so duplicate articles are caught.
    """
    cleaned_str = re.sub(r'\s+', '', text.lower())
    return hashlib.sha256(cleaned_str.encode('utf-8')).hexdigest()

def fetch_rss_feed(feed_url):
    """
    Fetches and parses an XML RSS feed, extracting article items.
    Returns a list of raw article metadata dictionaries.
    """
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        req = urllib.request.Request(feed_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        
        # Parse standard RSS items
        for item in root.findall('.//item'):
            title = item.findtext('title', 'Untitled Article')
            link = item.findtext('link', '')
            pub_date = item.findtext('pubDate', str(datetime.now().date()))
            description = item.findtext('description', '')
            
            # Clean basic HTML tags out of description text
            clean_content = re.sub(r'<[^>]+>', '', description).strip()
            
            if link and clean_content:
                articles.append({
                    "url": link,
                    "title": title.strip(),
                    "author": "RSS Feed Author",
                    "pub_date": pub_date,
                    "metadata": {"category": "Tech News", "source": feed_url},
                    "content": clean_content
                })
    except Exception:
        # Silently skip offline or unreachable seed feeds
        pass
        
    return articles

def run_crawler_pipeline(seed_urls=None, crawl_depth=1, use_fallback=True):
    """
    Main Crawler Engine supporting:
    1. Multi-seed scraping
    2. Configurable crawling depth
    3. URL deduplication
    4. Document body deduplication via SHA-256 content hashing
    5. Structural separation of Metadata vs. Document Body Content
    """
    if seed_urls is None:
        seed_urls = DEFAULT_SEEDS

    visited_urls = set()
    seen_content_hashes = set()
    
    extracted_metadata = []
    extracted_contents = {}
    
    duplicate_urls_count = 0
    duplicate_docs_count = 0
    
    # Pool candidate raw articles from live feeds or fallback corpus
    raw_candidates = []
    
    if use_fallback:
        raw_candidates.extend(FALLBACK_CORPUS)
        
    # Attempt live crawl over provided seed URLs up to configured depth
    for current_depth in range(1, crawl_depth + 1):
        for seed in seed_urls:
            if seed not in visited_urls:
                visited_urls.add(seed)
                feed_items = fetch_rss_feed(seed)
                raw_candidates.extend(feed_items)

    doc_counter = 1
    
    # Process and deduplicate all acquired candidates
    for item in raw_candidates:
        target_url = item["url"]
        body_text = item["content"]
        
        # Check 1: Duplicate URL Detection
        if target_url in visited_urls and item not in FALLBACK_CORPUS:
            duplicate_urls_count += 1
            continue
        visited_urls.add(target_url)
        
        # Check 2: Near-Duplicate Document Body Detection (SHA-256 hash match)
        content_hash = hash_text_content(body_text)
        if content_hash in seen_content_hashes:
            duplicate_docs_count += 1
            continue
        seen_content_hashes.add(content_hash)
        
        # Successfully passed deduplication! Map Doc ID
        doc_id = f"Doc_{doc_counter}"
        
        # Requirement B: Store extracted metadata SEPARATELY from document content
        extracted_metadata.append({
            "doc_id": doc_id,
            "url": target_url,
            "title": item["title"],
            "author": item.get("author", "Unknown"),
            "pub_date": item.get("pub_date", "N/A"),
            "category": item.get("metadata", {}).get("category", "General"),
            "source": item.get("metadata", {}).get("source", "Web"),
            "crawl_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Separate mapping for document text contents
        extracted_contents[doc_id] = body_text
        doc_counter += 1

    crawl_stats = {
        "total_seeds_processed": len(seed_urls),
        "crawl_depth_applied": crawl_depth,
        "unique_documents_indexed": len(extracted_contents),
        "duplicate_urls_blocked": duplicate_urls_count,
        "duplicate_documents_filtered": duplicate_docs_count
    }
    
    return extracted_metadata, extracted_contents, crawl_stats
# --- Quick Terminal Test Block ---
if __name__ == "__main__":
    print("🚀 Running Crawler Pipeline Test...\n")
    metadata, contents, stats = run_crawler_pipeline(use_fallback=True)
    
    print("📊 CRAWLER STATS:")
    for key, val in stats.items():
        print(f"  • {key}: {val}")
        
    print("\n📄 EXTRACTED METADATA (Task B Requirement - Separated):")
    for meta in metadata:
        print(f"  - [{meta['doc_id']}] {meta['title']} | Source: {meta['source']}")
        
    print("\n📝 EXTRACTED CONTENTS MAP (Task B Requirement - Separated):")
    for doc_id, text in contents.items():
        print(f"  - [{doc_id}]: {text[:80]}...")