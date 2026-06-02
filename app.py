"""
app.py  —  STEP 3: The chatbot itself (retrieve -> augment -> generate).
========================================================================

This is the part the user actually talks to. For each question it:

    1. RETRIEVES the most relevant chunks            (search.py)
    2. AUGMENTS a prompt by pasting those chunks in   (build_prompt below)
    3. GENERATES an answer with the LLM               (config.get_llm)

That 3-step loop is the whole idea of RAG: instead of hoping the model already
knows the answer, we *show* it the relevant text and ask it to answer from that.
This is what stops the model from making things up and lets it cite sources.

Run it with:   streamlit run app.py
"""

import streamlit as st

import config
from search import hybrid_search


# ---------------------------------------------------------------------------
# STEP 3a — The prompt: instructions + retrieved context + the question
# ---------------------------------------------------------------------------
# The prompt is just a big string we send to the LLM. The rules matter:
#   - "use ONLY the context" keeps answers grounded and reduces made-up facts.
#   - "say you don't know" is better than a confident wrong answer.
# Rewrite the wording to fit your domain and tone.
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say you don't know
rather than guessing. Be concise.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list) -> str:
    """Paste the retrieved chunks into the template as the 'context'."""
    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# STEP 3b — Answer one question end to end
# ---------------------------------------------------------------------------
def answer_question(question: str):
    """Retrieve -> build prompt -> ask the LLM. Returns (answer, chunks)."""
    chunks = hybrid_search(question)              # 1. RETRIEVE
    prompt = build_prompt(question, chunks)        # 2. AUGMENT
    llm = config.get_llm()
    response = llm.invoke(prompt)                  # 3. GENERATE
    return response.content, chunks


# ---------------------------------------------------------------------------
# STEP 3c — The Streamlit chat interface
# ---------------------------------------------------------------------------
# Streamlit reruns this whole file top-to-bottom on every interaction, so we
# keep the conversation in st.session_state (Streamlit's memory between reruns).
def main():
    st.title("📚 My RAG Chatbot")
    st.caption("Ask a question about the documents in the `data/` folder.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Re-draw the conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # The input box at the bottom. Returns the text when the user hits enter.
    question = st.chat_input("Type your question...")
    if question:
        # show + store the user's message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # get the answer and show it
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                answer, chunks = answer_question(question)
            st.markdown(answer)

            # Show the sources so the user can verify the answer. Transparency
            # is one of RAG's big advantages over a plain chatbot.
            with st.expander("Sources"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**{i}. {c['source']}**  _(via {c['method']} search)_")
                    st.caption(c["content"][:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
