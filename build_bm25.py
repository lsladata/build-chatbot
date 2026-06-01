"""
build_bm25.py
=============
Builds the BM25 keyword index that `legal_bm25_search.py` loads at runtime.

Run this once per document, AFTER you have produced the list of LangChain
`Document` objects in the ingestion notebook (the same `documents` list you
feed into Chroma). It writes `<docname>_bm25.pkl` with exactly the four keys
`LegalBM25Search` expects: bm25_index, doc_mapping, hierarchy_levels,
index_metadata.

Two deliberate design choices carried over from the reference implementation:

1. It indexes an *enhanced* string (hierarchy + source + page + content), not
   raw content, so structural metadata gets BM25 weight and a query like
   "chapter 5 deductible" can match on the heading as well as the body.
2. It KEEPS ALL STOPWORDS. For legal/technical text, words like "subject to",
   "in accordance with", "not", "shall" carry meaning, so removing them hurts
   precision. The query preprocessing in `legal_bm25_search.py` matches this.

Usage (from the notebook, after `documents` and `hierarchy_levels` exist):

    from build_bm25 import build_document_bm25
    build_document_bm25(documents, hierarchy_levels, docname="Texas Family Code")
    # -> writes bm25/Texas_Family_Code_bm25.pkl
"""

import os
import re
import pickle
from typing import List, Dict, Any

import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


def ensure_nltk_data():
    """Download the tokenizer data once if it isn't already present."""
    for path, name in [("tokenizers/punkt", "punkt"),
                       ("tokenizers/punkt_tab", "punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def preprocess_text_legal(text: str) -> List[str]:
    """
    Legal-specific preprocessing that PRESERVES ALL STOPWORDS.
    Only lowercases, normalizes whitespace, and drops punctuation-only tokens.
    This must stay identical to `LegalBM25Search.preprocess_query`.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    return tokens


def create_searchable_text(doc, hierarchy_levels: List[str]) -> str:
    """
    Combine document content with metadata for better searchability.
    Hierarchy is placed first (most specific first) so it gets higher BM25
    weight than the body text.
    """
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page", ""))

    # Most-specific hierarchy level first
    hierarchy_parts = []
    for level in reversed(hierarchy_levels):
        value = doc.metadata.get(level, "")
        if value:
            hierarchy_parts.append(value)

    hierarchy_text = " ".join(hierarchy_parts)
    return f"{hierarchy_text} {source} page {page} {doc.page_content}"


def tokenize_legal_documents(documents, enhanced_texts, hierarchy_levels):
    """
    Tokenize every document and build the parallel `doc_mapping`, which the
    search class returns to the app. Each entry keeps the original metadata,
    the raw content, the enhanced text, and a per-level hierarchy dict.
    """
    tokenized_corpus = []
    doc_mapping = []

    print(f"Tokenizing documents with hierarchy: {hierarchy_levels}")
    print("Preserving all stopwords for precision...")

    for i, (doc, enhanced_text) in enumerate(zip(documents, enhanced_texts)):
        tokens = preprocess_text_legal(enhanced_text)
        tokenized_corpus.append(tokens)

        doc_mapping.append({
            "doc_index": i,
            "metadata": doc.metadata.copy(),
            "content": doc.page_content,
            "enhanced_text": enhanced_text,
            "token_count": len(tokens),
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", ""),
            "hierarchy": {level: doc.metadata.get(level, "") for level in hierarchy_levels},
        })

        if i % 50 == 0:
            print(f"Processed {i + 1}/{len(documents)} documents")

    print("Tokenization complete.")
    return tokenized_corpus, doc_mapping


def build_and_save_legal_bm25(tokenized_corpus, doc_mapping, hierarchy_levels,
                              save_path: str = "legal_bm25_index") -> Dict[str, Any]:
    """
    Create the BM25 index package (with hierarchy stats) and pickle it.
    `save_path` is WITHOUT the .pkl extension.
    """
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    hierarchy_stats = {}
    for level in hierarchy_levels:
        unique_values = {doc["hierarchy"][level] for doc in doc_mapping
                        if doc["hierarchy"].get(level)}
        hierarchy_stats[level] = {
            "count": len(unique_values),
            "values": list(unique_values)[:10],  # small preview
        }

    index_package = {
        "bm25_index": bm25,
        "doc_mapping": doc_mapping,
        "hierarchy_levels": hierarchy_levels,
        "index_metadata": {
            "total_documents": len(tokenized_corpus),
            "average_doc_length": (sum(len(d) for d in tokenized_corpus)
                                   / len(tokenized_corpus)) if tokenized_corpus else 0,
            "sources": list({doc["source"] for doc in doc_mapping if doc["source"]}),
            "hierarchy_stats": hierarchy_stats,
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(index_package, f)

    print(f"\n✅ BM25 index saved to {save_path}.pkl")
    print(f"   • Total documents: {index_package['index_metadata']['total_documents']}")
    print(f"   • Hierarchy levels: {hierarchy_levels}")
    for level, stats in hierarchy_stats.items():
        print(f"   • {level.title()}s: {stats['count']}")

    return index_package


def build_document_bm25(documents, hierarchy_levels, docname: str,
                        out_dir: str = "bm25") -> Dict[str, Any]:
    """
    Convenience wrapper: enhanced text -> tokenize -> build -> save.

    The output filename matches what app.py loads:
        bm25/<docname with spaces -> underscores>_bm25.pkl

    Args:
        documents: list of LangChain Document objects (same list used for Chroma)
        hierarchy_levels: e.g. ['title','subtitle','chapter','subchapter','part','section']
                          (your metadata levels, excluding 'source' and 'page')
        docname: human-readable document name, e.g. "Texas Family Code"
    """
    ensure_nltk_data()
    enhanced_texts = [create_searchable_text(doc, hierarchy_levels) for doc in documents]
    print(f"Created {len(enhanced_texts)} enhanced texts.")

    tokenized_corpus, doc_mapping = tokenize_legal_documents(
        documents, enhanced_texts, hierarchy_levels
    )

    save_path = os.path.join(out_dir, f"{docname.replace(' ', '_')}_bm25")
    return build_and_save_legal_bm25(
        tokenized_corpus, doc_mapping, hierarchy_levels, save_path
    )
