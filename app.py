"""
Information Retrieval System - Master Streamlit Application
===========================================================
Integrates Crawling, Text Mining, Web Searching, PageRank/HITS, 
Recommendation Systems, and Evaluation Metrics into a unified Web UI.
Includes a 3-Tier Data Pipeline for bulletproof execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Import custom IR modules
from crawler import run_crawler_pipeline
from text_mining import TextMiningEngine, TextMiner
from web_search import WebSearchEngine
from recommender import RecommenderEngine
from evaluation import IREvaluator

# Define the RSS seeds directly here instead of importing them
RSS_SEEDS = [
    "http://export.arxiv.org/rss/cs.IR",  
    "http://export.arxiv.org/rss/cs.CL",  
    "http://export.arxiv.org/rss/cs.AI"   
]

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title=" Fullstack-IR | Assignment 2",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0rem !important;
        }
        .stMain, .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        [data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        header, [data-testid="stHeader"] {
            visibility: hidden !important;
            height: 0px !important;
            padding: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- APPLICATION TITLE & BANNER ---
st.title("🔎 FullStack-IR: A Streamlit Platform for Heterogeneous Data Ingestion, Search, Analytics & Recommendation")
st.caption("IR— Assignment 2 | A Streamlit-Driven End-to-End Information Retrieval Lifecycle Engine")

st.markdown("---")

# ==============================================================================
# 3-TIER DATA PIPELINE & ENGINE INITIALIZATION
# ==============================================================================
st.markdown("### 🗄️ Dataset Configuration")

with st.expander("🔍 What is happening behind the scenes? (3-Tier Data Pipeline)"):
    st.write("To guarantee zero downtime and reproducible evaluation metrics, this system uses a multi-tier data loading strategy:")
    st.markdown("""
    * **Tier 1 (Custom Upload):** The system prioritizes any custom `.csv` dataset you upload via the sidebar.
    * **Tier 2 (Live Web Crawl):** You can trigger a live scraper to fetch fresh web data dynamically.
    * **Tier 3 (Default Fallback):** If no file is uploaded and the live crawl fails, the system instantly falls back to a pre-loaded baseline `data.csv`.
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Expected CSV Schema for Custom Uploads")
    st.write("If you upload a custom dataset, please ensure it has the following column headers:")
    
    schema_df = pd.DataFrame({
        "Column Name": ["doc_id", "title", "source", "url", "content"],
        "Description": ["Unique ID (e.g., Doc_1)", "Document Title", "Origin (e.g., ArXiv)", "Reference Link", "Main body text to be indexed"]
    })
    st.dataframe(schema_df, hide_index=True, use_container_width=True)
    
    # Optional: Provide a downloadable template button
    sample_csv = "doc_id,title,source,url,content\nDoc_1,Sample Title,Manual,#,This is sample content."
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="template.csv",
        mime="text/csv"
    )

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("1. Upload Custom Dataset (CSV)", type=["csv"])
with col2:
    st.write("2. Or fetch fresh data")
    trigger_crawl = st.button("🌐 Trigger Live Web Crawl")

@st.cache_resource(show_spinner="Initializing Information Retrieval System...")
def initialize_ir_system(uploaded_buffer, crawl_triggered):
    """
    Executes crawler or loads CSV, then initializes all downstream IR engines once.
    Caches the state in RAM for fast Streamlit re-renders.
    """
    metadata_list = []
    contents_map = {}
    crawler_stats = {"duplicate_documents_filtered": 0}
    
    # Priority 1 & 3: CSV File Handling (Custom Upload or Default)
    if uploaded_buffer is not None or (not crawl_triggered and os.path.exists("data.csv")):
        file_to_read = uploaded_buffer if uploaded_buffer is not None else "data.csv"
        df = pd.read_csv(file_to_read)
        
        # Convert CSV DataFrame into the structures expected by your engines
        for idx, row in df.iterrows():
            doc_id = str(row.get("doc_id", f"Doc_{idx+1}"))
            contents_map[doc_id] = str(row.get("content", ""))
            metadata_list.append({
                "doc_id": doc_id,
                "title": str(row.get("title", f"Document {idx+1}")),
                "source": str(row.get("source", "CSV Upload")),
                "url": str(row.get("url", "#"))
            })
        crawler_stats["source_type"] = "CSV"
        
    # Priority 2: Live Crawl (or Fallback if CSV is missing)
    else:
        # Calls your custom crawler script
        metadata_list, contents_map, crawler_stats = run_crawler_pipeline(use_fallback=not crawl_triggered)
        crawler_stats["source_type"] = "Live Crawl"

    # Initialize Downstream Engines
    mining_engine = TextMiningEngine(contents_map)
    search_engine = WebSearchEngine(metadata_list, contents_map, mining_engine)
    recommender_engine = RecommenderEngine(metadata_list, mining_engine)
    evaluator_engine = IREvaluator()
    
    return {
        "metadata": metadata_list,
        "contents": contents_map,
        "crawler_stats": crawler_stats,
        "mining": mining_engine,
        "search": search_engine,
        "recommender": recommender_engine,
        "evaluator": evaluator_engine,
        "source": crawler_stats.get("source_type", "Unknown")
    }

# Load engines into state based on UI triggers
ir_system = initialize_ir_system(uploaded_file, trigger_crawl)

if ir_system["source"] == "CSV":
    st.success("✅ System successfully loaded from CSV dataset.")
else:
    st.info("🌐 System successfully loaded via Live Crawler / Web Engine.")

st.markdown("---")

# ==============================================================================
# SIDEBAR NAVIGATION MENU
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/330px-BITS_Pilani-Logo.svg.png", use_container_width=True)
st.sidebar.title("IR Navigation")

navigation_choice = st.sidebar.radio(
    "Select Workflow Module:",
    [
        "🏠 Overview Dashboard",
        "🕷️ Web Crawling",
        "📊 Text Mining & Indexing",
        "⚖️ Comparative Analysis (Task C)",
        "🔍 Search & PageRank",
        "💡 Recommender Panel",
        "📈 System Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Group Assignment 2**\n\nModules: NLTK, PageRank, HITS, Hybrid Recs, MAP/NDCG.")

# ==============================================================================
# TAB 1: OVERVIEW DASHBOARD
# ==============================================================================
if navigation_choice == "🏠 Overview Dashboard":
    st.header("🏠 System Overview Dashboard")
    st.write("Real-time telemetry and index metrics across the crawling and text processing pipelines.")
    
    corpus_stats = ir_system["mining"].get_corpus_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indexed Documents", len(ir_system["contents"]))
    col2.metric("Total Tokens Processed", corpus_stats.get("total_words_processed", 0))
    col3.metric("Vocabulary Size", corpus_stats.get("vocabulary_size", 0))
    col4.metric("Duplicates Filtered", ir_system["crawler_stats"].get("duplicate_documents_filtered", 0))

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📚 Corpus Category Breakdown")
        if "category_distribution" in corpus_stats:
            cat_df = pd.DataFrame(
                list(corpus_stats["category_distribution"].items()), 
                columns=["Category", "Document Count"]
            )
            st.bar_chart(cat_df.set_index("Category"))
        else:
            st.warning("Category data unavailable.")
        
    with col_b:
        st.subheader("🔥 Top Corpus Keywords")
        if "top_frequent_words" in corpus_stats:
            top_words_df = pd.DataFrame(
                corpus_stats["top_frequent_words"], 
                columns=["Term", "Frequency"]
            )
            st.dataframe(top_words_df, use_container_width=True, height=280)
        else:
            st.warning("Keyword data unavailable.")

# ==============================================================================
# TAB 2: WEB CRAWLING INTERFACE
# ==============================================================================
elif navigation_choice == "🕷️ Web Crawling":
    st.header("🕷️ Web Crawling & Content Acquisition")
    st.write("Configure RSS seed URLs, crawling depth, and explore the extracted document metadata.")
    
    with st.expander("⚙️ Crawler Configuration Options", expanded=True):
        selected_seeds = st.multiselect("Active Seed Sources:", RSS_SEEDS, default=RSS_SEEDS)
        crawl_depth = st.slider("Crawl Depth Horizon:", min_value=1, max_value=3, value=1)
        st.info("To trigger a live crawl, use the button at the top of the application.")

    st.subheader("📄 Extracted Document Metadata Repository")
    metadata_df = pd.DataFrame(ir_system["metadata"])
    st.dataframe(metadata_df, use_container_width=True)

# ==============================================================================
# TAB 3: TEXT MINING & INDEXING
# ==============================================================================
elif navigation_choice == "📊 Text Mining & Indexing":
    st.header("📊 Text Preprocessing & Vector Mining")
    st.write("Explore NLTK tokenization, vocabulary distributions, and document TF-IDF profiles.")
    
    selected_doc_id = st.selectbox("Select Document to Inspect:", list(ir_system["contents"].keys()))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📝 Raw Text Content ({selected_doc_id})")
        st.info(ir_system["contents"][selected_doc_id])
        
    with col2:
        st.subheader("🏷️ Top TF-IDF Keywords")
        keywords = ir_system["mining"].extract_top_keywords(selected_doc_id, top_n=8)
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Weight"])
        st.dataframe(kw_df, use_container_width=True)
        
        doc_category = ir_system["mining"].profile_and_classify_document(selected_doc_id)
        st.success(f"**Assigned Category:** {doc_category}")

# ==============================================================================
# TAB 4: COMPARATIVE ANALYSIS (NEW - TASK C)
# ==============================================================================
elif navigation_choice == "⚖️ Comparative Analysis (Task C)":
    st.header("⚖️ Comparative Analysis of Preprocessing (Task C)")
    st.write("Compare how different normalization and vectorization strategies affect the vocabulary size and processing time.")

    # Extract raw documents from the initialized system
    raw_documents = list(ir_system["contents"].values())

    miner = TextMiner()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strategy A")
        norm_a = st.selectbox("Normalization A", ['stemming', 'lemmatization'], key='norm_a')
        vec_a = st.selectbox("Vectorization A", ['tfidf', 'count'], key='vec_a')
        
    with col2:
        st.subheader("Strategy B")
        norm_b = st.selectbox("Normalization B", ['lemmatization', 'stemming'], key='norm_b')
        vec_b = st.selectbox("Vectorization B", ['count', 'tfidf'], key='vec_b')

    if st.button("Run Comparative Analysis"):
        with st.spinner("Processing Strategy A..."):
            results_a = miner.build_features(raw_documents, norm_strategy=norm_a, vectorizer_type=vec_a)
        
        with st.spinner("Processing Strategy B..."):
            results_b = miner.build_features(raw_documents, norm_strategy=norm_b, vectorizer_type=vec_b)
            
        st.write("### Results")
        comparison_df = pd.DataFrame({
            "Metric": ["Vocabulary Size (Features)", "Processing Time (Seconds)"],
            f"Strategy A ({norm_a} + {vec_a})": [results_a['vocab_size'], f"{results_a['time_taken']:.4f}"],
            f"Strategy B ({norm_b} + {vec_b})": [results_b['vocab_size'], f"{results_b['time_taken']:.4f}"]
        })
        
        st.table(comparison_df)
        
        st.info("**Observation:** Notice how Lemmatization generally produces a slightly different vocabulary size than Stemming, and CountVectorizer differs from TF-IDF in feature weighting (though they share the same vocabulary if normalization is identical).")

# ==============================================================================
# TAB 5: SEARCH & PAGERANK
# ==============================================================================
elif navigation_choice == "🔍 Search & PageRank":
    st.header("🔍 Intelligent Search & Web Graph Ranking")
    st.write("Execute queries using score fusion combining TF-IDF relevance and PageRank/HITS authority.")
    
    search_query = st.text_input("Enter Search Keywords:", "ai model framework")
    use_pr = st.checkbox("Enable PageRank Authority Boosting (0.7 Relevance + 0.3 PageRank)", value=True)
    
    if search_query:
        results = ir_system["search"].execute_search(search_query, apply_pagerank_boost=use_pr, top_k_results=10)
        
        st.subheader(f"Search Results ({len(results)} matches)")
        
        for idx, res in enumerate(results, 1):
            with st.container():
                st.markdown(f"### {idx}. {res['title']}")
                st.caption(f"**Doc ID:** {res['doc_id']} | **Source:** {res['source']} | **Fused Rank Score:** {res['final_rank_score']:.4f}")
                st.write(res["snippet"])
                st.markdown("---")

    st.subheader("🕸️ Top PageRank & HITS Graph Authorities")
    col_pr, col_hits = st.columns(2)
    
    with col_pr:
        st.markdown("**Top PageRank Scores**")
        top_pr = sorted(ir_system["search"].pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        pr_df = pd.DataFrame(top_pr, columns=["Doc ID", "PageRank Score"])
        st.dataframe(pr_df, use_container_width=True)
        
    with col_hits:
        st.markdown("**Top HITS Authority Scores**")
        top_auth = sorted(ir_system["search"].authority_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        auth_df = pd.DataFrame(top_auth, columns=["Doc ID", "Authority Score"])
        st.dataframe(auth_df, use_container_width=True)

# ==============================================================================
# TAB 6: RECOMMENDER PANEL
# ==============================================================================
elif navigation_choice == "💡 Recommender Panel":
    st.header("💡 Personalized Recommender Engine")
    st.write("Generate Content-Based, Collaborative, or Hybrid recommendations for any article.")
    
    col_sel, col_slider = st.columns([1, 2])
    with col_sel:
        target_doc = st.selectbox("Choose Base Article:", list(ir_system["contents"].keys()), index=0)
        rec_type = st.radio("Recommendation Strategy:", ["Hybrid", "Content-Based", "Collaborative Filtering"])
        
    with col_slider:
        alpha = st.slider("Content Weight (α vs 1-α Collaborative):", 0.0, 1.0, 0.6)
        
    st.subheader("📌 Recommended Articles")
    
    if rec_type == "Content-Based":
        recs = ir_system["recommender"].get_content_based_recommendations(target_doc, top_n=5)
        rec_df = pd.DataFrame(recs)
    elif rec_type == "Collaborative Filtering":
        recs = ir_system["recommender"].get_collaborative_recommendations(target_user_id=0, top_n=5)
        rec_df = pd.DataFrame(recs)
    else:
        recs = ir_system["recommender"].get_hybrid_recommendations(target_doc, target_user_id=0, alpha=alpha, top_n=5)
        rec_df = pd.DataFrame(recs)
        
    st.dataframe(rec_df, use_container_width=True)

# ==============================================================================
# TAB 7: SYSTEM EVALUATION
# ==============================================================================
elif navigation_choice == "📈 System Evaluation":
    st.header("📈 Evaluation Metrics & IR Benchmarking")
    st.write("Measure Precision, Recall, MAP, MRR, and NDCG@5 against ground-truth sets.")
    
    # Ground truth evaluation simulation controls
    retrieved_sim = ["Doc_1", "Doc_29", "Doc_3", "Doc_28", "Doc_12", "Doc_8"]
    ground_truth_sim = ["Doc_1", "Doc_3", "Doc_12"]
    
    st.subheader("🎯 Evaluation Telemetry Dashboard")
    report = ir_system["evaluator"].generate_full_evaluation_report(retrieved_sim, ground_truth_sim)
    
    metrics_df = pd.DataFrame(list(report.items()), columns=["IR Metric", "Calculated Score"])
    
    col_metrics, col_chart = st.columns([1, 2])
    
    with col_metrics:
        st.dataframe(metrics_df, use_container_width=True, height=310)
        
    with col_chart:
        st.bar_chart(metrics_df.set_index("IR Metric"))