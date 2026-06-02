"""
ingest.py  —  STEP 1: Turn your documents into searchable indexes.
==================================================================

You run this ONCE (and again whenever your documents change). It reads every
file in the docs folder and builds the two things the chatbot searches:

    1. A VECTOR STORE  (for "semantic" search — meaning-based)
    2. A KEYWORD INDEX (for "BM25" search — exact word/number based)

Why both? They fail in opposite ways, so together they cover each other:
  - Semantic search is great at "find the bit about cancelling my plan" even if
    the document says "termination of coverage" — but it can miss exact terms.
  - Keyword search nails exact terms, names, and numbers ("Section 4.2",
    "deductible") but is blind to paraphrasing.

Run it with:   python ingest.py
"""

import os
import re
import glob
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ---------------------------------------------------------------------------
# STEP 1a — LOAD raw text from each file in the docs folder
# ---------------------------------------------------------------------------
def load_documents(folder: str):
    """
    Read every .txt and .pdf in `folder` and return a list of LangChain
    `Document` objects. A Document is just text plus a little "metadata"
    dictionary (here: which file it came from). Metadata is what lets us later
    tell the user *where* an answer came from.
    """
    documents = []

    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        name = os.path.basename(path)

        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        elif path.lower().endswith(".pdf"):
            # PyMuPDF (imported as fitz) reads text out of a PDF, page by page.
            import fitz
            text = ""
            with fitz.open(path) as pdf:
                for page in pdf:
                    text += page.get_text()
        else:
            continue  # skip anything that isn't .txt or .pdf

        # `source` is the document title; we'll show it as the citation.
        documents.append(Document(page_content=text, metadata={"source": name}))
        print(f"  loaded {name}  ({len(text):,} characters)")

    return documents


# ---------------------------------------------------------------------------
# STEP 1b — SPLIT each document into small overlapping chunks
# ---------------------------------------------------------------------------
def split_documents(documents):
    """
    LLMs answer best from small, focused pieces of text, and search is more
    precise on small pieces too. So we cut each document into ~CHUNK_SIZE-char
    chunks. RecursiveCharacterTextSplitter tries to break on paragraph and
    sentence boundaries first, so chunks stay readable instead of being cut
    mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  split into {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# STEP 1c — BUILD the vector store (semantic search)
# ---------------------------------------------------------------------------
def build_vector_store(chunks):
    """
    For each chunk we ask the embedding model for its vector, then store all the
    vectors in Chroma (a small local vector database saved to disk). At question
    time we embed the question the same way and ask Chroma for the nearest
    chunks. THIS STEP CALLS THE EMBEDDING API, so it needs your keys and costs a
    little money.
    """
    from langchain_chroma import Chroma

    print("  embedding chunks and writing the vector store... (uses the API)")
    Chroma.from_documents(
        documents=chunks,
        embedding=config.get_embeddings(),
        persist_directory=config.VECTOR_DIR,
    )
    print(f"  vector store saved to {config.VECTOR_DIR}")


# ---------------------------------------------------------------------------
# STEP 1d — BUILD the keyword index (BM25)
# ---------------------------------------------------------------------------
def _tokenize(text: str):
    """
    Turn text into a clean list of lowercase words. BM25 compares the *words* in
    the question to the *words* in each chunk, so both must be tokenized the
    same way (see search.py, which reuses this exact function).
    """
    text = re.sub(r"\s+", " ", text.lower()).strip()
    # keep only alphanumeric tokens (drops punctuation)
    return [tok for tok in re.split(r"\W+", text) if tok]


def build_keyword_index(chunks):
    """
    BM25 is a classic keyword-ranking algorithm (think: a smarter Ctrl+F that
    ranks results). It needs no API and no keys — it just counts word matches,
    weighting rare words more. We tokenize every chunk, build the index, and
    pickle it together with the chunk text + metadata so search.py can return
    the original text later.
    """
    from rank_bm25 import BM25Okapi

    tokenized_chunks = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    package = {
        "bm25": bm25,
        # the original chunks, in the SAME order as tokenized_chunks, so a BM25
        # result index maps straight back to its text + source
        "chunks": [{"content": c.page_content, "metadata": c.metadata} for c in chunks],
    }

    os.makedirs(os.path.dirname(config.KEYWORD_PATH), exist_ok=True)
    with open(config.KEYWORD_PATH, "wb") as f:
        pickle.dump(package, f)
    print(f"  keyword index saved to {config.KEYWORD_PATH}")


# ---------------------------------------------------------------------------
# Run all four sub-steps in order
# ---------------------------------------------------------------------------
def main():
    print("STEP 1a: loading documents...")
    documents = load_documents(config.DOCS_FOLDER)
    if not documents:
        print(f"No .txt or .pdf files found in '{config.DOCS_FOLDER}/'. Add some and rerun.")
        return

    print("STEP 1b: splitting into chunks...")
    chunks = split_documents(documents)

    print("STEP 1c: building the vector store...")
    build_vector_store(chunks)

    print("STEP 1d: building the keyword index...")
    build_keyword_index(chunks)

    print("\n✅ Done. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
