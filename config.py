"""
config.py  —  One place for all settings and the AI clients.
============================================================

WHY THIS FILE EXISTS
--------------------
Every other file needs two things: (1) your settings (folder paths, model
names) and (2) the two AI clients — one that turns text into vectors
("embeddings") and one that writes answers ("the LLM"). Instead of repeating
that setup in every file, we do it ONCE here and import it everywhere else.

Keeping secrets here also means there is exactly ONE place to look when
something is misconfigured.

HOW SECRETS WORK
----------------
We never write API keys in code. Instead we read them from "environment
variables", which we load from a local file called `.env` (copy `.env.example`
to `.env` and fill it in). `.env` is listed in `.gitignore`, so it never gets
uploaded to GitHub.
"""

import os
from dotenv import load_dotenv

# Reads the .env file and puts its values into os.environ so we can read them
# below. Call this once, as early as possible.
load_dotenv()


# ---------------------------------------------------------------------------
# 1. SETTINGS YOU CAN CHANGE
# ---------------------------------------------------------------------------
# These are plain values, safe to edit. They control where data lives and how
# the pipeline behaves. No secrets here.

# Folder containing the documents you want the chatbot to read (.txt or .pdf).
DOCS_FOLDER = "data"

# Where the two search indexes get written by ingest.py.
#   - the vector store (semantic search)
#   - the keyword index (BM25)
VECTOR_DIR = "storage/vectors"     # Chroma writes a small database here
KEYWORD_PATH = "storage/keyword_index.pkl"

# How big each chunk of text should be, in characters. RAG works on small
# pieces of a document, not the whole thing. ~1000 chars (~150-200 words) is a
# sensible default: big enough to hold an idea, small enough to be specific.
CHUNK_SIZE = 1000
# How much neighbouring chunks overlap. Overlap stops us from cutting a
# sentence in half and losing the meaning at the boundary.
CHUNK_OVERLAP = 150

# How many chunks to retrieve for each question before answering.
TOP_K = 4


# ---------------------------------------------------------------------------
# 2. SECRETS (read from .env — never hard-code these)
# ---------------------------------------------------------------------------
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "")


# ---------------------------------------------------------------------------
# 3. THE TWO AI CLIENTS
# ---------------------------------------------------------------------------
# We wrap each client in a small function so it is only created when needed
# (creating it checks your keys, which we don't want to do just by importing).

def get_embeddings():
    """
    Returns the EMBEDDING model: it converts a piece of text into a list of
    numbers (a "vector") that captures its meaning. Two texts about similar
    things get similar vectors, which is how semantic search finds relevant
    chunks even when the wording differs.
    """
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        openai_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_version=AZURE_API_VERSION,
        azure_deployment=EMBEDDING_DEPLOYMENT,
    )


def get_llm():
    """
    Returns the CHAT model (the LLM): it reads the retrieved chunks plus the
    user's question and writes the final answer in plain language.
    """
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        openai_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_version=AZURE_API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0,   # 0 = factual and repeatable; higher = more creative
    )


# ---------------------------------------------------------------------------
# WANT A DIFFERENT PROVIDER?  (e.g. plain OpenAI instead of Azure)
# ---------------------------------------------------------------------------
# Only these two functions are provider-specific. To switch to OpenAI, replace
# the bodies above with:
#
#     from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#     def get_embeddings():
#         return OpenAIEmbeddings(model="text-embedding-3-large")
#     def get_llm():
#         return ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# (and put OPENAI_API_KEY in your .env). Nothing else in the project changes.
