# IR Assignment 2 - Advanced Web Crawling, PageRank, & Hybrid Recommendations

**Group Number:** Group 39  

### 👥 Team Members & Contributions
* **Suresh Kumar** * **ID:** 2025aa05159 | **Email:** 2025aa05159@wilp.bits-pilani.ac.in[cite: 2]
* **Sayyad Mohd Abulasar Mohd Abuzafar Qaiser** * **ID:** 2025aa05147 | **Email:** 2025aa05147@wilp.bits-pilani.ac.in[cite: 2]
* **Jitendra Rajput** * **ID:** 2025aa05145 | **Email:** 2025aa05145@wilp.bits-pilani.ac.in[cite: 2]

## 1. Problem Statement
The goal of this assignment is to extend foundational Information Retrieval concepts into a complex, web-oriented ecosystem. This involves:
1. **Web Crawling:** Building an RSS-driven crawler capable of extracting, parsing, and deduplicating live articles.
2. **Text Mining & Indexing:** Generating TF-IDF vector representations and vocabulary profiles using NLTK.
3. **Web Search & Link Analysis:** Implementing PageRank and HITS algorithms to evaluate document authority, combined with TF-IDF for fused search relevance.
4. **Recommender System:** Engineering a Hybrid Recommender engine that fuses Content-Based (Cosine Similarity) and Collaborative Filtering scores.
5. **System Evaluation:** Computing standard IR benchmarking metrics (Precision, Recall, MAP, MRR, NDCG@5) against ground-truth datasets.
6. **UI Integration:** Deploying the entire multi-engine architecture into an interactive, zero-latency web interface via Streamlit.

**Source & Data Pipeline:** 
The system employs a robust 3-Tier Data Pipeline to ensure continuous availability in restricted lab environments:
1. **Custom Upload:** Prioritizes user-uploaded `.csv` datasets via the Streamlit UI.
2. **Live Web Crawling:** Dynamically fetches from ArXiv RSS Feeds (cs.IR, cs.CL, cs.AI) if triggered.
3. **Static Fallback:** Defaults to a bundled `data.csv` corpus if no upload is provided and live crawling fails.

**Format & Processing:** 
Regardless of the source (CSV or Live Web), documents are normalized into unified in-memory metadata dictionaries (Titles, URLs, Source) mapped to raw textual content. The ingestion pipeline rigorously filters out duplicate documents before passing the clean data into the text mining engine.
## 3. Project Structure
This repository maintains a fully decoupled, modular architecture:

```text
IR_ASSIGNMENT-2/
│
├── app.py                  # Master Streamlit dashboard integrating all backend engines
├── crawler.py              # Web crawling, RSS parsing, and document deduplication (Task A & B)
├── text_mining.py          # TF-IDF vectorization and NLP categorization (Task C)
├── web_search.py           # Inverted index, PageRank, HITS, and Score Fusion (Task D)
├── recommender.py          # Content-based, Collaborative, and Hybrid recommendations (Task E)
├── evaluation.py           # Performance metrics: Precision, Recall, MAP, NDCG, MRR (Task F)
├── requirements.txt        # Virtual lab environment external package declarations
└── README.md               # Code deployment and evaluation documentation


## 4. Architectural Highlights

- **Fused Search Scoring:** Queries don't just rely on keyword matching (TF-IDF); they are multiplied by the synthetic network graph's PageRank score to ensure authoritative documents rank higher.
- **Hybrid Recommendations:** Users can tweak an alpha (α) weight slider in the UI to dynamically adjust the balance between Content-Based features and simulated User-Collaborative histories.
- **In-Memory Caching:** The Streamlit `@st.cache_resource` decorator ensures that the crawler and PageRank matrices only compute once upon boot, resulting in instant UI state updates.



## How to Run Locally

**1. Install dependencies:**
```bash
pip install -r requirements.txt

**2. Run the Streamlit App::**
```bash
python -m streamlit run app.py
