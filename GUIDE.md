# 📘 Step-by-Step Guide: Build Your Own RAG Chatbot

This guide walks you from an empty folder to a working chatbot over **your own documents**. It assumes basic Python familiarity. Read [`README.md`](./README.md) first for the big picture.

**Contents**

1. [Prerequisites](#1-prerequisites)
2. [Set up your environment & keys](#2-set-up-your-environment--keys)
3. [Describe your document structure](#3-describe-your-document-structure)
4. [Ingest your documents](#4-ingest-your-documents)
5. [Build the BM25 keyword index](#5-build-the-bm25-keyword-index)
6. [Run the chatbot](#6-run-the-chatbot)
7. [The four retrieval modes, in code](#7-the-four-retrieval-modes-in-code)
8. [Using a different LLM provider](#using-a-different-llm-provider)
9. [Deploying](#9-deploying)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

- **Python 3.10+**
- An **LLM provider** with both a chat model and an embedding model. The reference uses **Azure OpenAI** (`gpt-3.5-turbo-16k` for chat, `text-embedding-3-large` for embeddings), but any provider works (see [provider swap](#using-a-different-llm-provider)).
- Your documents as **PDFs** (or adapt the loader for `.txt`, `.docx`, HTML, etc.).
- *(Optional)* an **Airtable** base if you want interaction logging.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Set up your environment & keys

Copy the example file and fill in real values:

```bash
cp .env.example .env
```

```dotenv
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_CHAT_DEPLOYMENT=your-chat-deployment-name
AZURE_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name

# Optional logging
AIRTABLE_API_TOKEN=pat...
AIRTABLE_BASE_ID=app...
AIRTABLE_TABLE_NAME=My_Log
```

Both `app.py` and the notebook call `load_dotenv()`, so these become available via `os.environ`. **`.env` is git-ignored** — your keys never get committed.

> 🔐 **Golden rule:** secrets only ever live in `.env` (local) or `st.secrets` (deployed). Never paste a key into a `.py` or `.ipynb` file.

---

## 3. Describe your document structure

This is the step that makes retrieval *smart* instead of generic. Open **`metadata_fields.py`**.

RAG quality depends on the model seeing **complete, well-scoped context**. If your documents have a hierarchy (chapters, sections, clauses…), capturing it as metadata lets you:

- **Filter** retrieval ("only search Chapter 5"), and
- **Expand** a small matched chunk back to its full parent section before answering.

You describe the hierarchy with LangChain `AttributeInfo` objects, ordered **from broadest to narrowest**:

```python
from langchain.chains.query_constructor.schema import AttributeInfo

# Example: a generic structured document
my_doc_metadata = [
    AttributeInfo(name="source",  description="Document title",            type="string"),
    AttributeInfo(name="chapter", description="Top-level chapter heading",  type="string"),
    AttributeInfo(name="section", description="Section within a chapter",   type="string"),
    AttributeInfo(name="page",    description="Page number in the source",  type="integer"),
]

# Map every document name to its schema
METADATA_FIELDS = {
    "My Handbook": my_doc_metadata,
    "Another Doc": my_doc_metadata,
}
```

**Real examples** of different shapes (from the reference app):

```python
# A legal code: Title > Subtitle > Chapter > Subchapter > Part > Section
standard_code_metadata = [
    AttributeInfo(name="source",     description="Legal document source title", type="string"),
    AttributeInfo(name="title",      description="Major division",              type="string"),
    AttributeInfo(name="subtitle",   description="Subdivision under title",      type="string"),
    AttributeInfo(name="chapter",    description="Chapter division",            type="string"),
    AttributeInfo(name="subchapter", description="Subchapter division",         type="string"),
    AttributeInfo(name="part",       description="Part division",               type="string"),
    AttributeInfo(name="section",    description="Individual legal section",    type="string"),
    AttributeInfo(name="page",       description="Page number",                 type="integer"),
]

# A rules document with a flatter shape: Rule > Subdivision
rules_metadata = [
    AttributeInfo(name="source", description="The source document title", type="string"),
    AttributeInfo(name="rule",   description="The main rule heading",     type="string"),
    AttributeInfo(name="page",   description="Page number",               type="integer"),
]
```

> **Rule of thumb:** keep `source` first and `page` last. The levels in between are your hierarchy, broadest → narrowest. The app uses the *second-to-last* level as the default unit for "expand to full section."

If your documents are flat (no headings), you can use just `source` and `page` — the hybrid search still works, you just lose the structure-aware expansion.

---

## 4. Ingest your documents

Open **`notebooks/build_database.ipynb`**. The pipeline is:

> **PDF → extract text (with position/font) → detect headings → build `Document`s with metadata → semantic chunking → Chroma vector store**

### 4.1 Extract text with layout info

The reference uses **PyMuPDF (`fitz`)**, which gives you each text span *plus its font and position* — that's how it distinguishes headings from body text:

```python
import fitz
import pandas as pd

pdf = fitz.open(f"./pdfs/{name}.pdf")

rows = []
for i in range(pdf.page_count):
    for block in pdf[i].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                rows.append({
                    "Text": span["text"],
                    "Page": i,
                    "Vertical Pos": span["origin"][1],
                    "Horizontal Pos": span["origin"][0],
                    "Font": span["font"],
                    "Font Size": span["size"],
                })
df = pd.DataFrame(rows).sort_values(["Page", "Vertical Pos", "Horizontal Pos"])
```

### 4.2 Detect your headings

This is the part **you must adapt to your documents**. The reference detects legal headings by font (`Courier-Bold`) plus regex (`TITLE 1.`, `CHAPTER 5.`, `Sec. 1.002.`…). Your headings might be:

- **Font-based** — bold/large text is a heading (like the reference).
- **Numbering-based** — lines matching `^\d+\.\d+` are sections.
- **Already structured** — if your source is Markdown/HTML, the heading tags do the work for you.

The detector returns *which hierarchy level* a line is and *its label*, and you carry a "current context" dict so every chunk inherits the headings above it:

```python
hierarchy_levels = ["chapter", "section"]   # match your metadata_fields.py (minus source/page)
current_context = dict.fromkeys(hierarchy_levels)

documents, buffer, current_page = [], [], 0
for _, row in df.iterrows():
    level, label = detect_heading(row)        # <-- YOUR logic here
    if level:                                 # a heading: flush the buffer, update context
        if buffer:
            documents.append(Document(
                page_content="\n".join(buffer),
                metadata={"source": name, "page": current_page,
                          **{k: v for k, v in current_context.items() if v}},
            ))
            buffer = []
        current_context[level] = label
        for lower in hierarchy_levels[hierarchy_levels.index(level) + 1:]:
            current_context[lower] = None     # reset narrower levels
    else:                                     # body text: accumulate
        buffer.append(row["Text"])
        current_page = row["Page"]
```

### 4.3 Semantic chunking

Rather than fixed-size chunks, the reference splits where the *meaning* shifts, using LangChain's `SemanticChunker`:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
import os

embeddings = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
    model="text-embedding-3-large",
)

splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
docs = splitter.split_documents(documents)
```

> Semantic chunking calls the embedding API, so it costs money and time. For a cheaper start, swap in `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` — no API calls — and upgrade later.

### 4.4 Build the vector store

```python
from langchain_chroma import Chroma

knowledge_base = Chroma.from_documents(
    documents=docs,
    persist_directory=f"./embeddings/{name}.pdf",   # app.py reads from here
    embedding=embeddings,
)
```

Repeat 4.1–4.4 for each document. The reference wraps it all in a `SemanticEmbeddings(name, start_page, edit)` function so you can loop over a whole library.

---

## 5. Build the BM25 keyword index

`legal_bm25_search.py` **loads** a pre-built index; the matching builder is **`build_bm25.py`**. Run it once per document, right after step 4, passing the same `documents` list you fed into Chroma:

```python
from build_bm25 import build_document_bm25

# documents       : the list of LangChain Documents from step 4
# hierarchy_levels : your metadata levels, broadest -> narrowest, WITHOUT source/page
#                    e.g. ['title','subtitle','chapter','subchapter','part','section']
build_document_bm25(documents, hierarchy_levels, docname="Texas Family Code")
# -> writes bm25/Texas_Family_Code_bm25.pkl   (the path app.py loads)
```

That's the whole call. The builder produces a pickle with exactly the four keys
`LegalBM25Search.load_index()` reads back — `bm25_index`, `doc_mapping`,
`hierarchy_levels`, `index_metadata` — so the search class works unchanged.

**Two design choices in `build_bm25.py` worth understanding**, because they're what make the keyword side actually good for structured documents:

1. **It indexes *enhanced* text, not raw content.** Before tokenizing, each chunk is turned into `"<hierarchy> <source> page <n> <content>"` (most-specific heading first). That gives the structure real BM25 weight, so a query like *"chapter 5 deductible"* can match on the heading as well as the body.
2. **It keeps every stopword on purpose.** For legal/technical text, words like *not*, *shall*, *subject to*, *in accordance with* change the meaning, so the usual "strip stopwords" step would hurt precision. The query preprocessing in `legal_bm25_search.py` is identical (lowercase → normalize whitespace → drop punctuation-only tokens), which is essential — **the index and the query must be preprocessed the same way** or scores break.

If your documents are flat (no headings), pass `hierarchy_levels=[]` and it still works — you just index `source + page + content`.

> ⚠️ A pickle file executes code on load. Only ever load `.pkl` files **you built yourself**. Never load one from an untrusted source.

---

## 6. Run the chatbot

```bash
streamlit run app.py
```

Then in the UI:

1. Pick a category and a document in the sidebar (or tick **Enable multi-select** to load several).
2. Type a question.
3. Read the answer, expand **Sources** to verify, and leave 👍/👎 feedback.

`app.py` wires everything together:

- `get_embeddings()` / `get_llm()` — cached provider clients.
- `get_knowledge_base()` — loads the Chroma store for a document.
- `LegalBM25Search(...)` — loads the BM25 index.
- `SelfQueryRetriever` — lets the LLM translate questions into metadata filters automatically.
- `PROMPT_TEMPLATE` — the instructions that keep answers grounded in sources. **Rewrite this for your domain.**

---

## 7. The four retrieval modes, in code

All four live in `app.py` as `handle_*_search_with_costs(...)`. They build on each other:

**Michelangelo — pure vector search**
```python
answer, sources = process_query_embeddings(question, doc_name)   # RetrievalQA over Chroma
```

**Raphael — hybrid (vector + BM25)**
```python
emb, bm25 = run_both_searches_concurrent(question, retriever, searcher)  # run in parallel
merged = interleave_search_results(emb, dic_to_doc(bm25))                # alternate results
enhanced = enhance_documents_dynamic(merged, metadata_attrs)             # prepend metadata header
answer, tokens = process_query_with_cost_tracking(question, enhanced)
```

**Leonardo — hybrid + hierarchy expansion** *(the default)*
```python
emb, bm25 = run_both_searches_concurrent(question, retriever, searcher)
emb = enhance_embeddings_with_hierarchy_dynamic(emb, searcher, metadata_attrs)  # expand each
merged = interleave_search_results(emb, dic_to_doc(bm25))                       # hit to its
enhanced = enhance_documents_dynamic(merged, metadata_attrs)                    # full section
answer, tokens = process_query_with_cost_tracking(question, enhanced, llm)
```

**Donatello — Leonardo + LLM re-ranking**
```python
# ...same retrieval as Leonardo, then:
filtered, filter_tokens = filter_documents_by_relevance_with_cost_tracking(
    question, enhanced_docs, llm
)   # asks the LLM to score each chunk 0-100 and keep only the relevant ones
answer, tokens = process_query_with_cost_tracking(question, filtered)
```

**Which should you use?** Start with **Michelangelo**. If exact terms get missed, go **Raphael**. If answers feel fragmentary, go **Leonardo**. If retrieval is noisy and you can afford an extra LLM call, go **Donatello**.

---

## Using a different LLM provider

The only provider-specific code is in three client constructors. To use, say, standard OpenAI instead of Azure, change the imports and constructors — the rest of the pipeline is identical.

```python
# Azure (reference)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
llm = AzureChatOpenAI(azure_deployment="...", azure_endpoint="...", openai_api_key="...")
emb = AzureOpenAIEmbeddings(deployment="...", azure_endpoint="...", openai_api_key="...")

# OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"])

# Anthropic for chat (keep an embedding provider for the vectors)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=os.environ["ANTHROPIC_API_KEY"])
```

If you switch chat models, update the pricing table in **`cost_tracker.py`** (`self.pricing`) so cost tracking stays accurate, and update the `chat_model=` strings in the `handle_*` functions.

---

## 9. Deploying

**Streamlit Community Cloud** is the quickest path. Two differences from local:

1. Put secrets in the app's **Secrets** UI (TOML), not in `.env`:
   ```toml
   AZURE_OPENAI_API_KEY = "sk-..."
   AZURE_OPENAI_ENDPOINT = "https://..."
   ```
   `os.environ` won't see these, so read them with `st.secrets` — or bridge them once at startup:
   ```python
   import os, streamlit as st
   for k, v in st.secrets.items():
       os.environ.setdefault(k, str(v))
   ```
2. Some hosts ship an old `sqlite3` that Chroma rejects. Uncomment the three lines at the very top of `app.py` and add `pysqlite3-binary` to `requirements.txt`.

Your built `embeddings/` and `bm25/` folders must be present in the deploy (they're git-ignored by default). Either commit them deliberately for a small corpus, or host them and download at startup.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|--------|--------------|-----|
| `KeyError` on a document name | `metadata_fields.py` has no entry for it | Add the document to `METADATA_FIELDS`. |
| `FileNotFoundError: BM25 index not found` | The `.pkl` wasn't built or the path/name differs | Re-run the BM25 builder; the app expects `bm25/<Doc_Name>_bm25.pkl` with spaces as underscores. |
| Chroma / `sqlite3` error on deploy | Host's sqlite is too old | Enable the `pysqlite3` shim (see [Deploying](#9-deploying)). |
| Empty or irrelevant answers | Chunks too small or headings mis-detected | Check the notebook's heading detection; try Leonardo mode; inspect the **Sources** expander. |
| Costs look wrong | Pricing table is stale | Update `self.pricing` in `cost_tracker.py`. |
| Auth errors | Keys missing from env | Confirm `.env` exists and `load_dotenv()` runs before the clients are built. |

---

That's the whole loop: **describe your structure → ingest → index → serve**. Start with one document and Michelangelo mode, confirm it works end to end, then add documents and richer retrieval. Good luck! 🚀
