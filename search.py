"""
search.py  —  STEP 2: Find the chunks most relevant to a question.
==================================================================

This file is the "R" in RAG (Retrieval). Given a question, it finds the most
relevant chunks from the indexes that ingest.py built. The app then hands those
chunks to the LLM to write an answer.

There are three functions, building up in capability:

    vector_search()   -> meaning-based     (needs the embedding API)
    keyword_search()  -> exact-word based  (no API, instant)
    hybrid_search()   -> combines both     (recommended)

Start by reading vector_search and keyword_search; hybrid_search just merges
their results.
"""

import pickle

import config
from ingest import _tokenize   # reuse the EXACT same tokenizer used at build time


# ---------------------------------------------------------------------------
# SEMANTIC (vector) search
# ---------------------------------------------------------------------------
def vector_search(question: str, top_k: int = None):
    """
    1. Embed the question into a vector (same model used in ingest.py).
    2. Ask Chroma for the chunks whose vectors are closest to it.
    Closeness in vector space ≈ closeness in meaning, so this finds relevant
    chunks even when they share no words with the question.

    Returns a list of dicts: {"content": ..., "source": ..., "method": "vector"}
    """
    from langchain_chroma import Chroma

    top_k = top_k or config.TOP_K

    store = Chroma(
        persist_directory=config.VECTOR_DIR,
        embedding_function=config.get_embeddings(),
    )
    hits = store.similarity_search(question, k=top_k)

    return [
        {"content": h.page_content,
         "source": h.metadata.get("source", "unknown"),
         "method": "vector"}
        for h in hits
    ]


# ---------------------------------------------------------------------------
# KEYWORD (BM25) search
# ---------------------------------------------------------------------------
def keyword_search(question: str, top_k: int = None):
    """
    Loads the keyword index and ranks every chunk by how well its words match
    the question's words (rare words count for more). No API call — this works
    even with no keys, which makes it a great way to test ingestion quickly.

    Returns the same dict shape as vector_search, with "method": "keyword".
    """
    top_k = top_k or config.TOP_K

    with open(config.KEYWORD_PATH, "rb") as f:
        package = pickle.load(f)
    bm25 = package["bm25"]
    chunks = package["chunks"]

    # Score the question against the whole corpus, then take the best `top_k`.
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue   # 0 means "no shared words" — not relevant, skip it
        results.append({
            "content": chunks[i]["content"],
            "source": chunks[i]["metadata"].get("source", "unknown"),
            "method": "keyword",
        })
    return results


# ---------------------------------------------------------------------------
# HYBRID search  (recommended)
# ---------------------------------------------------------------------------
def hybrid_search(question: str, top_k: int = None):
    """
    Run BOTH searches and interleave the results (1st vector, 1st keyword,
    2nd vector, 2nd keyword, ...). Interleaving is a simple, robust way to give
    each method equal say without having to make their scores comparable (vector
    "distances" and BM25 "scores" are on totally different scales, so we rank by
    POSITION instead of raw score).

    We also drop duplicates: if both methods return the same chunk, we keep it
    once. That overlap is actually a good sign — both methods agreeing means the
    chunk is probably very relevant.
    """
    top_k = top_k or config.TOP_K

    vec = vector_search(question, top_k)
    key = keyword_search(question, top_k)

    merged = []
    seen = set()
    for i in range(max(len(vec), len(key))):
        for source_list in (vec, key):
            if i < len(source_list):
                item = source_list[i]
                # de-duplicate on the first 100 chars of the chunk
                fingerprint = item["content"][:100]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    merged.append(item)

    return merged[:top_k]


# Quick manual test:  python search.py "your question here"
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "what is the refund policy?"
    print(f"Question: {q}\n")
    # keyword_search needs no API key, so it's the safest smoke test:
    for r in keyword_search(q):
        print(f"[{r['method']}] {r['source']}: {r['content'][:120]}...")
