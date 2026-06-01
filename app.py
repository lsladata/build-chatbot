#https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/6
# Needed for azure but not needed for local:
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from metadata_fields import METADATA_FIELDS
import os
from datetime import datetime
from dotenv import load_dotenv

# Load secrets from a local .env file (see .env.example)
load_dotenv()

import time
import requests
from pyairtable import Api  
import json

from legal_bm25_search import LegalBM25Search

import concurrent.futures
from typing import Tuple, List, Dict, Any

from langchain_core.documents import Document
import base64
import numpy as np
from cost_tracker import AzureOpenAICostTracker, InteractionCostLogger
import html

# Airtable 
AIRTABLE_CONFIG = {
    # Pulled from environment / .env  ->  never commit real tokens
    'api_token': os.environ.get("AIRTABLE_API_TOKEN", ""),
    'base_id': os.environ.get("AIRTABLE_BASE_ID", ""),
    'table_name': os.environ.get("AIRTABLE_TABLE_NAME", "Legal_Juris_Log"),
}

#Create the required fields in Airtable table:
#required_fields = { 'Timestamp': 'DateTime', 'Document Name': 'SingleLineText', 'Question': 'LongText', 
#'Answer': 'LongText', 'Search Mode': 'SingleSelect', 'Session ID': 'SingleLineText', 
#'Answer Length': 'Number', 'Source Count': 'Number' }

@st.cache_resource 
def get_airtable_client(): 
    """Initialize Airtable client with caching""" 
    try: 
        api = Api(AIRTABLE_CONFIG['api_token']) 
        table = api.table(AIRTABLE_CONFIG['base_id'], 
        AIRTABLE_CONFIG['table_name']) 
        return table 
    except Exception as e: 
        st.error(f"Failed to initialize Airtable client: {str(e)}") 
        return None

def log_to_airtable_with_costs(
    question: str, 
    answer: str, 
    doc_name: str, 
    search_mode: str = "Unknown",
    response_time: float = 0.0, 
    source_count: int = 0,
    cost_data: dict = None,  # NEW PARAMETER
    thumbs_up: str = "", 
    thumbs_down: str = "", 
    survey_response: str = "", 
    other_response: str = ""
):
    """
    Enhanced version of log_to_airtable with comprehensive cost tracking
    
    Args:
        cost_data: Dictionary containing cost breakdown from InteractionCostLogger
                  Expected keys: 'chat_model', 'chat_prompt_tokens', 'chat_completion_tokens',
                                'chat_total_tokens', 'chat_input_cost', 'chat_output_cost',
                                'chat_total_cost', 'embedding_model', 'embedding_tokens',
                                'embedding_cost', 'documents_count', 'extra_llm_cost',
                                'extra_llm_prompt_tokens', 'extra_llm_completion_tokens',
                                'total_cost', 'cost_breakdown'
    """
    try:
        url = f"https://api.airtable.com/v0/{AIRTABLE_CONFIG['base_id']}/{AIRTABLE_CONFIG['table_name']}"
        
        headers = {
            'Authorization': f"Bearer {AIRTABLE_CONFIG['api_token']}",
            'Content-Type': 'application/json'
        }
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        timestamp = datetime.now().isoformat()
        
        # Build basic fields (existing)
        fields = {
            'Timestamp': timestamp,
            'Document': doc_name,
            'Question': question[:1000],
            'Answer': answer[:2000], 
            'Search Mode': search_mode,
            'Session ID': st.session_state.session_id,
            'Response Time Seconds': response_time,
            'Source Count': source_count,
            'Answer Length': len(answer)
        }
        
        # Add cost tracking fields if cost_data is provided
        if cost_data:
            fields.update({
                # Chat model costs
                'Chat Model': cost_data.get('chat_model', ''),
                'Chat Prompt Tokens': cost_data.get('chat_prompt_tokens', 0),
                'Chat Completion Tokens': cost_data.get('chat_completion_tokens', 0),
                'Chat Total Tokens': cost_data.get('chat_total_tokens', 0),
                'Chat Input Cost': cost_data.get('chat_input_cost', 0),
                'Chat Output Cost': cost_data.get('chat_output_cost', 0),
                'Chat Total Cost': cost_data.get('chat_total_cost', 0),
                
                # Embedding costs
                'Embedding Model': cost_data.get('embedding_model', ''),
                'Embedding Tokens': cost_data.get('embedding_tokens', 0),
                'Embedding Cost': cost_data.get('embedding_cost', 0),
                'Documents Retrieved': cost_data.get('documents_count', 0),
                
                # Extra LLM costs (Donatello)
                'Extra LLM Cost': cost_data.get('extra_llm_cost', 0),
                'Extra LLM Prompt Tokens': cost_data.get('extra_llm_prompt_tokens', 0),
                'Extra LLM Completion Tokens': cost_data.get('extra_llm_completion_tokens', 0),
                
                # Total cost
                'Total Cost': cost_data.get('total_cost', 0),
                
                # Cost breakdown percentages
                'Chat Cost Percentage': f"{cost_data.get('cost_breakdown', {}).get('chat_percentage', 0)}%",
                'Embedding Cost Percentage': f"{cost_data.get('cost_breakdown', {}).get('embedding_percentage', 0)}%",
                'Extra Cost Percentage': f"{cost_data.get('cost_breakdown', {}).get('extra_percentage', 0)}%"
            })

        data = {
            'records': [
                {
                    'fields': fields
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            response_data = response.json()
            record_id = response_data['records'][0]['id']
            return True, record_id
        else:
            st.error(f"Airtable error: {response.status_code} - {response.text}")
            return False, None
            
    except Exception as e:
        st.error(f"Airtable logging error: {str(e)}")
        return False, None


def update_airtable_feedback(record_id: str, thumbs_up: str = "", thumbs_down: str = "", 
                           survey_response: str = "", other_response: str = ""):
    """Enhanced feedback update with separate Other response tracking"""
    try:
        url = f"https://api.airtable.com/v0/{AIRTABLE_CONFIG['base_id']}/{AIRTABLE_CONFIG['table_name']}/{record_id}"
        
        headers = {
            'Authorization': f"Bearer {AIRTABLE_CONFIG['api_token']}",
            'Content-Type': 'application/json'
        }
        
        # Build fields - only update fields with values
        fields = {}
        
        if thumbs_up:
            fields['Thumbs Up'] = thumbs_up
        if thumbs_down:
            fields['Thumbs Down'] = thumbs_down
        if survey_response:
            fields['Survey Response'] = survey_response
        if other_response:  # New field
            fields['Other_Survey_Response'] = other_response
        
        data = {
            'fields': fields
        }
        
        response = requests.patch(url, headers=headers, json=data, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        st.error(f"Feedback update error: {str(e)}")
        return False

def display_feedback_buttons(question_id: str, record_id: str):
    """Simplified feedback buttons"""
    
    feedback_status = st.session_state.get(f'feedback_{question_id}', None)
    survey_submitted = st.session_state.get(f'survey_submitted_{question_id}', False)
    
    if feedback_status == 'thumbs_up':
        st.success("👍 Glad you found this answer helpful - Thank you!")
        return
    elif feedback_status == 'thumbs_down' and survey_submitted:
        st.info("👎 Thank you for your feedback - we're working to improve!")
        return
    elif feedback_status == 'thumbs_down' and not survey_submitted:
        display_feedback_survey(question_id)
        return
    
    col1, col2, col3 = st.columns([1, 1, 12])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{question_id}", help="This answer was helpful"):
            st.session_state[f'feedback_{question_id}'] = 'thumbs_up'
            
            # Just update Airtable - no chat history changes
            success = update_airtable_feedback(record_id, thumbs_up="Yes")
            if success:
                st.success("✅ Thanks for your positive feedback!")
            else:
                st.error("❌ Failed to save feedback")
    
    with col2:
        if st.button("👎", key=f"thumbs_down_{question_id}", help="This answer needs improvement"):
            st.session_state[f'feedback_{question_id}'] = 'thumbs_down'

def display_feedback_survey(question_id: str):
    """Enhanced survey with better state management"""
    
    # Track form state separately from submission state
    form_key = f"feedback_form_{question_id}"
    
    st.markdown("---")
    st.markdown("### Help us improve - What went wrong?")
    
    with st.form(key=form_key):
        survey_options = [
            "Wrong Answer",
            "Hallucination", 
            "Sources Incorrect",
            "Too Slow",
            "Other"
        ]
        
        selected_issues = st.multiselect(
            "Select all that apply:",
            survey_options
        )
        
        other_text = ""
        if "Other" in selected_issues:
            other_text = st.text_area(
                "Please specify what went wrong:",
                placeholder="Tell us specifically what went wrong...",
                help="This will help us understand and fix the issue"
            )
        
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if not selected_issues:
                st.warning("Please select at least one option.")
                # Don't return - let them try again
            elif "Other" in selected_issues and not other_text.strip():
                st.warning("Please specify what went wrong in the text area.")
                # Don't return - let them try again
            else:
                # Valid submission - process it
                survey_response = ", ".join(selected_issues)
                other_response = other_text.strip() if "Other" in selected_issues and other_text.strip() else ""
                
                record_id = st.session_state.get('current_record_id')
                
                if record_id:
                    success = update_airtable_feedback(
                        record_id, 
                        thumbs_down="Yes",
                        survey_response=survey_response,
                        other_response=other_response
                    )
                    
                    if success:
                        st.success("✅ Thank you for your feedback!")
                        if other_response:
                            st.info("📝 Your specific feedback has been recorded and will help us improve.")
                        # Only mark as submitted on success
                        st.session_state[f'survey_submitted_{question_id}'] = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to submit feedback")
            

#Set Title for the tab on browsers window
st.set_page_config(
    layout="wide",
    page_title="Legal Juris", 
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
#if 'chat_history' not in st.session_state:
#    st.session_state.chat_history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'multi_select_mode' not in st.session_state:
    st.session_state.multi_select_mode = False
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []
if 'loaded_docs' not in st.session_state:
    st.session_state.loaded_docs = []
if 'multi_doc_resources' not in st.session_state:
    st.session_state.multi_doc_resources = {}

# Configuration - Consider using st.secrets for production
AZURE_CONFIG = {
    # Pulled from environment / .env  ->  never commit real keys
    'api_key': os.environ.get("AZURE_OPENAI_API_KEY", ""),
    'endpoint': os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    'api_version': os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
}

# Initialize cost tracking
cost_tracker = AzureOpenAICostTracker()
interaction_logger = InteractionCostLogger(cost_tracker)





# Optimized prompt template
PROMPT_TEMPLATE = """You are a legal research assistant specializing in Texas law.

Answer the user's question using ONLY the provided sources and metadata.
Do NOT use outside legal knowledge, assumptions, or general background information.
If the answer is not fully supported by the sources, say so explicitly.


## Task
Review the question and all provided sources.
Use the source text plus metadata (source name, jurisdiction, authority type, part, section, rule, date, court, etc.) to determine which authorities are most relevant.


## Rules for legal analysis
1. Prefer sources that directly answer the question over general background sources.
2. Treat metadata as evidence of authority, scope, and hierarchy, but not as a substitute for source text.
3. When possible, quote the exact controlling language from the most relevant source.
4. If multiple sources are relevant, synthesize them carefully.
5. If sources conflict, identify the conflict and explain each position.
6. If the question is ambiguous, answer the narrowest reasonable interpretation and briefly state the ambiguity.
7. If the sources are insufficient, state exactly what is missing.

## Citation rules
1. Cite only the source(s) that directly support the proposition.
2. Quote exact controlling language when it materially affects the answer.
3. Do NOT cite a source unless it is also listed in Sources Used.

## Output format
Answer:
Write 2 short paragraphs in plain English, legally precise, no more than 300 words total.

Sources Used:

Source Name | Hierarchical Path

Source Name | Hierarchical Path

Source Name | Hierarchical Path

If the sources are insufficient, add:
Missing Information:
Briefly state what additional authority, source text, or facts would be needed.

Question:
{question}

Available Sources:
{context}
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

@st.cache_resource(show_spinner="Loading embeddings...")
def get_embeddings():
    """Cached embeddings initialization"""
    return AzureOpenAIEmbeddings(
        openai_api_type="azure",
        openai_api_key=AZURE_CONFIG['api_key'],
        azure_endpoint=AZURE_CONFIG['endpoint'],
        deployment="model3large",
        model="text-embedding-3-large",
        api_version=AZURE_CONFIG['api_version'],
        max_retries=3
    )

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_knowledge_base(doc_name: str):
    """Cached knowledge base loading with error handling"""
    try:
        embeddings = get_embeddings()
        knowledge_base = Chroma(
            persist_directory=f"./embeddings/{doc_name}.pdf",
            embedding_function=embeddings
        )
        return knowledge_base
    except Exception as e:
        st.error(f"Error loading knowledge base for {doc_name}: {str(e)}")
        return None

@st.cache_resource(show_spinner="Initializing AI model...")
def get_llm():
    """Cached LLM initialization"""
    return AzureChatOpenAI(
        azure_deployment="Chat3-5-16",
        azure_endpoint=AZURE_CONFIG['endpoint'],
        openai_api_key=AZURE_CONFIG['api_key'],
        api_version=AZURE_CONFIG['api_version'],
        temperature=0.1,
        request_timeout=30,
        max_retries=3
    )

@st.cache_resource(show_spinner="Loading knowledge base...", ttl=300)
def load_embeddings(doc_name: str) -> Tuple[str, List]:
    """Enhanced query processing with metadata-aware search"""
    try:
        knowledge_base = get_knowledge_base(doc_name)
        if not knowledge_base:
            return "Error: Could not load knowledge base", []
        
        llm = get_llm()
        
        metadata_field_info = METADATA_FIELDS[doc_name]
        return knowledge_base, llm, metadata_field_info
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

@st.cache_resource
def get_retriever(doc_name: str, _llm, _knowledge_base, _metadata_field_info):
    """Cache the retriever setup with all dependencies"""
    return SelfQueryRetriever.from_llm(
        llm=_llm,
        vectorstore=_knowledge_base,
        document_contents="Legal document text with hierarchical structure",
        metadata_field_info=_metadata_field_info,
        search_type="similarity",
        enable_limit=True
    )


@st.cache_data(ttl=300)
def process_query_embeddings(question: str, doc_name: str) -> Tuple[str, List]:
    """Enhanced query processing with metadata-aware search"""
    try:
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        response = qa.invoke({"query": question})
        return response["result"], response["source_documents"]
        
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")


@st.cache_resource
def initialize_document_resources(doc_name: str):
    """Initialize all document-related resources at once"""
    try:
        # Get metadata
        metadata_attrs = METADATA_FIELDS.get(doc_name, [])
        if metadata_attrs is None:
            st.error(f"No metadata schema found for: {doc_name}")
            st.write("Available metadata keys:", list(METADATA_FIELDS.keys()))
            return None
        # Load embeddings and models
        knowledge_base = get_knowledge_base(doc_name)
        llm = get_llm()
        
        # Create retriever
        retriever = get_retriever(doc_name, llm, knowledge_base, metadata_attrs)
        
        return {
            'metadata_attrs': metadata_attrs,
            'knowledge_base': knowledge_base,
            'llm': llm,
            'retriever': retriever,
        }
    except Exception as e:
        st.error(f"Error initializing resources: {e}")
        return None



def process_query_with_cost_tracking(question: str, combined_docs: List, llm_obj) -> Tuple[str, List, Dict]:
    """
    Enhanced version of process_query that tracks token usage and costs
    
    Returns:
        Tuple of (answer, sources, cost_info)
        cost_info contains: prompt_tokens, completion_tokens, total_tokens
    """
    # Create QA chain with your custom prompt
    qa_chain = load_qa_chain(
        llm=llm_obj,
        chain_type="stuff",
        prompt=PROMPT
    )

    # Get answer - the response includes token usage in Langchain
    result = qa_chain({"input_documents": combined_docs, "question": question})
    answer = result["output_text"]
    
    # Extract token usage from LLM result
    # Langchain stores this in the llm_output field
    token_usage = {}
    if hasattr(llm_obj, 'last_llm_output') and llm_obj.last_llm_output:
        token_usage = llm_obj.last_llm_output.get('token_usage', {})
    
    # If not available, estimate from text
    if not token_usage:
        # Estimate prompt tokens
        context_text = "\n".join([doc.page_content for doc in combined_docs])
        prompt_text = PROMPT_TEMPLATE.format(question=question, context=context_text)
        prompt_tokens = cost_tracker.count_tokens(prompt_text)
        completion_tokens = cost_tracker.count_tokens(answer)
        
        token_usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'estimated': True  # Flag to indicate this was estimated
        }
    
    return answer, token_usage


@st.cache_data(show_spinner="Processing question...non-embeddings", ttl=300)
def source_search(question: str, retriever_obj) -> Tuple[str, List]:
    """Enhanced query processing with metadata-aware search"""
    try:
        response_source = retriever_obj.get_relevant_documents(question)
        return response_source
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        return []


def run_both_searches_concurrent(user_question: str, retriever_obj, searcher_obj) -> Tuple[Tuple[str, List], List[Dict[str, Any]]]:
    """
    Run both BM25 and embeddings search simultaneously.
    
    Args:
        user_question: The search query
        doc_name: Document name for embeddings search
        bm25_index_path: Path to BM25 index file
        top_k: Number of top results for BM25 search
    
    Returns:
        Tuple containing (embeddings_result, bm25_results)
    """
    
    def run_embeddings_search():
        """Wrapper for embeddings search"""
        try:
            return source_search(user_question, retriever_obj)
        except Exception as e:
            print(f"Embeddings search error: {e}")
            return f"Error: {str(e)}", []
    
    def run_bm25_search():
        """Wrapper for BM25 search"""
        try:
            return searcher_obj.search(user_question, 3)
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []
    
    # Run both searches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        embeddings_future = executor.submit(run_embeddings_search)
        bm25_future = executor.submit(run_bm25_search)
        
        # Wait for both to complete and get results
        embeddings_result = embeddings_future.result()
        bm25_results = bm25_future.result()
    
    return embeddings_result, bm25_results


@st.cache_data(ttl=300)
def dic_to_doc(json_doc):
    jsondoc=[]
    for jdoc in json_doc:
        jsondoc.append(Document(
            page_content=jdoc["content"],
            metadata=jdoc["metadata"]
            ))
    return jsondoc

def get_primary_hierarchy_level(metadata_attrs):
    """
    Get the most appropriate hierarchy level for enhancement based on document structure.
    Returns the second-most-granular level (excluding 'page').
    
    For example:
    - Texas Family Code: 'section' (below 'part')
    - Texas Rules of Disciplinary Procedure: 'section' (below 'part')
    - Texas Rules of Civil Procedure: 'rule' (below 'sub-rule')
    """
    # Get all attribute names excluding 'source' and 'page'
    hierarchy_levels = [
        attr.name for attr in metadata_attrs 
        if attr.name not in ['source', 'page']
    ]
    
    if len(hierarchy_levels) >= 2:
        # Return second-to-last level (more specific than top level, but not too granular)
        return hierarchy_levels[-2]
    elif len(hierarchy_levels) == 1:
        # Only one level exists, use it
        return hierarchy_levels[0]
    else:
        # Fallback (shouldn't happen)
        return None


def get_best_available_hierarchy_level(doc_metadata, metadata_attrs):
    """
    Find the most specific hierarchy level that actually exists in the document.
    Searches from most specific to least specific.
    """
    # Get hierarchy levels (excluding 'source' and 'page')
    hierarchy_levels = [
        attr.name for attr in metadata_attrs 
        if attr.name not in ['source', 'page']
    ]
    
    # Check from most specific to least specific (reverse order)
    for level in reversed(hierarchy_levels):
        if level in doc_metadata and doc_metadata[level]:
            return level
    
    # Return first level as fallback
    return hierarchy_levels[0] if hierarchy_levels else None

def render_plain_wrapped_text(text: str):
    """Render literal text with wrapping and preserved line breaks."""
    escaped = html.escape(text or "").replace("$", "&#36;")
    st.markdown(
        f"""
        <div style="
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            line-height: 1.5;
            font-family: inherit;
        ">{escaped}</div>
        """,
        unsafe_allow_html=True
    )

def display_sources_with_tabs(sources: List, doc_name: str):
    """Efficiently display sources in tabs with metadata"""
    if not sources:
        st.info("No sources found for this query.")
        return

    metadata_attrs = METADATA_FIELDS.get(doc_name, [])

    tab_labels = [f"Source {i+1}" for i in range(len(sources))]

    with st.expander(f"Sources from: {doc_name}", expanded=False):
        if len(sources) == 1:
            display_single_source(sources[0], doc_name, metadata_attrs)
        else:
            tabs = st.tabs(tab_labels)
            for index, (tab, source) in enumerate(zip(tabs, sources)):
                with tab:
                    display_single_source(source, doc_name, metadata_attrs, index + 1)


def display_single_source(source, doc_name: str, metadata_attrs, source_num: int = 1):
    """Display a single source with formatted metadata"""
    ordered_fields = [attr.name for attr in metadata_attrs] if metadata_attrs else list(source.metadata.keys())
    
    col1, col2 = st.columns([1, 2])

    with col1:
        for field in ordered_fields:
            if hasattr(source, 'metadata') and field in source.metadata:
                value = source.metadata[field]
                if value:
                    formatted_value = format_metadata_value(field, value)
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {formatted_value}")

    with col2:
        st.markdown("**Content:**")
        content = source.page_content if hasattr(source, 'page_content') else str(source)

        if content.startswith('[') and ']\n' in content:
            content = content.split(']\n', 1)[1]

        render_plain_wrapped_text(content)

def format_metadata_value(field: str, value) -> str:
    """Format metadata values for better display"""
    if field == "source":
        return f"*{value}*"
    elif field == "page":
        # Handle page number formatting
        try:
            page_num = int(value) + 1 if isinstance(value, (int, str)) and str(value).isdigit() else value
            return f"*{page_num}*"
        except:
            return f"*{value}*"
    else:
        return f"*{str(value).title()}*"


def update_response_time_in_airtable(record_id: str, response_time: float):
    """Update the Airtable record with the actual response time after display"""
    try:
        url = f"https://api.airtable.com/v0/{AIRTABLE_CONFIG['base_id']}/{AIRTABLE_CONFIG['table_name']}/{record_id}"
        
        headers = {
            'Authorization': f"Bearer {AIRTABLE_CONFIG['api_token']}",
            'Content-Type': 'application/json'
        }
        
        data = {
            'fields': {
                'Response Time Seconds': response_time
            }
        }
        
        response = requests.patch(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            # Optional: Show debug info
            # st.success(f"✅ Updated response time: {response_time}s")
            return True
        else:
            st.error(f"Failed to update response time: {response.status_code}")
            return False
        
    except Exception as e:
        st.error(f"Response time update error: {str(e)}")
        return False

def colored_info(message, color_theme="blue"):
    themes = {
        "blue": {"bg": "#e8f4fd", "border": "#1f77b4", "text": "#1f77b4"},
        "red": {"bg": "#fef2f2", "border": "#ef4444", "text": "#dc2626"},
        "orange": {"bg": "#fff7ed", "border": "#f97316", "text": "#ea580c"},
        "purple": {"bg": "#f3e8ff", "border": "#a855f7", "text": "#7c3aed"}
    }
    
    theme = themes.get(color_theme, themes["blue"])
    
    st.markdown(f"""
    <div style="
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: {theme['bg']};
        border-left: 4px solid {theme['border']};
        color: {theme['text']};
        margin: 1rem 0;
        font-weight: 500;
    ">
    {message}
    </div>
    """, unsafe_allow_html=True)

def interleave_search_results(embeddings_results: list, bm25_results: list) -> list:
    """
    Interleave embeddings and BM25 results alternately.
    
    Args:
        embeddings_results: List of embedding search results
        bm25_results: List of BM25 search results
        
    Returns:
        List with alternating results from both sources
    """
    bm25_count = len(bm25_results)
    emb_count = len(embeddings_results)
    all_doc = []
    
    for i in range(0, max(emb_count, bm25_count)):
        if i < emb_count:
            all_doc.append(embeddings_results[i])
        if i < bm25_count:
            all_doc.append(bm25_results[i])
    
    return all_doc

def enhance_embeddings_with_hierarchy_dynamic(embeddings_results, searcher, metadata_attrs):
    """
    Dynamically enhance embeddings based on document structure.
    
    Args:
        embeddings_results: Initial embedding search results
        searcher: BM25 searcher instance
        metadata_attrs: Metadata attributes for the current document
        
    Returns:
        Enhanced embeddings results as Document objects
    """
    if not embeddings_results:
        return []
    emb_results_full = []
    
    for meta in embeddings_results:
        # Find the best hierarchy level for THIS specific document
        hierarchy_lvl = get_best_available_hierarchy_level(meta.metadata, metadata_attrs)
        
        if hierarchy_lvl and hierarchy_lvl in meta.metadata:
            search_term = meta.metadata[hierarchy_lvl]
            emb_results_full.append(searcher.find_by_any_level(
                search_term=search_term,
                level=hierarchy_lvl,
                exact_match=True
            ))
        else:
            # Keep original if no suitable hierarchy level found
            emb_results_full.append([{
                "content": meta.page_content,
                "metadata": meta.metadata
            }])

    list_of_emb_results = []
    for emb_results in emb_results_full:
        if emb_results:
            list_of_emb_results.append(emb_results[0])
    
    return dic_to_doc(list_of_emb_results)

def handle_michelangelo_search_with_costs(user_question: str, doc_name: str) -> tuple:
    """Enhanced Michelangelo with cost tracking"""
    # Call your existing embeddings search (modify process_query_embeddings to return tokens)
    # For now, we'll wrap the existing function
    answer, sources = process_query_embeddings(user_question, doc_name)
    
    # Estimate costs since we don't have direct token access from cached function
    # This is approximate - for exact tracking, you'd need to modify process_query_embeddings
    prompt_estimate = cost_tracker.count_tokens(user_question) + \
                     cost_tracker.estimate_document_tokens(sources)
    completion_estimate = cost_tracker.count_tokens(answer)
    
    cost_log = interaction_logger.log_interaction(
        search_mode="Michelangelo",
        question=user_question,
        answer=answer,
        chat_model="gpt-3.5-turbo-16k",
        embedding_model="text-embedding-3-large",
        prompt_tokens=prompt_estimate,
        completion_tokens=completion_estimate,
        documents_retrieved=sources,
        extra_llm_calls=None
    )
    
    return answer, sources, cost_log


def handle_raphael_search_with_costs(user_question: str, doc_name: str, metadata_attrs) -> tuple:
    """Enhanced Raphael with cost tracking"""
    embeddings_results, json_bm25_results = run_both_searches_concurrent(
        user_question=user_question,
        doc_name=doc_name
    )
    
    bm25_results = dic_to_doc(json_bm25_results)
    all_doc = interleave_search_results(embeddings_results, bm25_results)
    
    enhanced_docs = enhance_documents_dynamic(all_doc, metadata_attrs)
    answer, token_usage = process_query_with_cost_tracking(user_question, enhanced_docs)
    sources = all_doc
    
    cost_log = interaction_logger.log_interaction(
        search_mode="Raphael",
        question=user_question,
        answer=answer,
        chat_model="gpt-3.5-turbo-16k",
        embedding_model="text-embedding-3-large",
        prompt_tokens=token_usage.get('prompt_tokens', 0),
        completion_tokens=token_usage.get('completion_tokens', 0),
        documents_retrieved=sources,
        extra_llm_calls=None
    )
    
    return answer, sources, cost_log


def handle_leonardo_search_with_costs(user_question: str, doc_name: str, retriever, searcher, metadata_attrs, llm_obj) -> tuple:
    """Enhanced Leonardo with cost tracking"""
    embeddings_results, json_bm25_results = run_both_searches_concurrent(
        user_question=user_question,
        retriever_obj=retriever,
        searcher_obj=searcher
    )

    if embeddings_results is None:
        embeddings_results = []

    if json_bm25_results is None:
        json_bm25_results = []

    bm25_results = dic_to_doc(json_bm25_results) if json_bm25_results else []

    embeddings_results = enhance_embeddings_with_hierarchy_dynamic(
        embeddings_results, searcher, metadata_attrs
    )

    all_doc = interleave_search_results(embeddings_results, bm25_results)

    enhanced_docs = enhance_documents_dynamic(all_doc, metadata_attrs)
    answer, token_usage = process_query_with_cost_tracking(user_question, enhanced_docs, llm_obj)
    sources = all_doc

    cost_log = interaction_logger.log_interaction(
        search_mode="Leonardo",
        question=user_question,
        answer=answer,
        chat_model="gpt-3.5-turbo-16k",
        embedding_model="text-embedding-3-large",
        prompt_tokens=token_usage.get('prompt_tokens', 0),
        completion_tokens=token_usage.get('completion_tokens', 0),
        documents_retrieved=sources,
        extra_llm_calls=None
    )

    return answer, sources, cost_log



def handle_multi_doc_leonardo_search_with_costs(user_question: str, multi_doc_resources: Dict[str, Dict]) -> tuple:
    """Run Leonardo across multiple loaded documents, merge sources, then answer once."""
    first_resource = next(iter(multi_doc_resources.values()))
    llm_obj = first_resource["llm"]
    all_sources = []
    all_enhanced_docs = []

    def search_single_doc(item):
        doc_name, resources = item

        try:
            searcher = resources["searcher"]
            metadata_attrs = resources["metadata_attrs"]
            retriever = resources["retriever"]

            embeddings_results, json_bm25_results = run_both_searches_concurrent(
                user_question=user_question,
                retriever_obj=retriever,
                searcher_obj=searcher
            )

            if embeddings_results is None:
                embeddings_results = []

            if json_bm25_results is None:
                json_bm25_results = []

            bm25_results = dic_to_doc(json_bm25_results) if json_bm25_results else []

            embeddings_results = enhance_embeddings_with_hierarchy_dynamic(
                embeddings_results, searcher, metadata_attrs
            )

            all_doc = interleave_search_results(embeddings_results, bm25_results)

            for doc in all_doc:
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc.metadata["loaded_doc_name"] = doc_name

            enhanced_docs = enhance_documents_dynamic(all_doc, metadata_attrs)

            return doc_name, all_doc[:4], enhanced_docs[:4], None

        except Exception as e:
            return doc_name, [], [], str(e)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(search_single_doc, item)
            for item in multi_doc_resources.items()
        ]

        failed_docs = []

        for future in concurrent.futures.as_completed(futures):
            doc_name, doc_sources, doc_enhanced_docs, error_msg = future.result()

            if error_msg:
                failed_docs.append(f"{doc_name}: {error_msg}")
                continue

            all_sources.extend(doc_sources)
            all_enhanced_docs.extend(doc_enhanced_docs)

    if not all_enhanced_docs:
        answer = "I couldn't find relevant content in the loaded documents for that question."

        cost_log = interaction_logger.log_interaction(
            search_mode="Leonardo",
            question=user_question,
            answer=answer,
            chat_model="gpt-3.5-turbo-16k",
            embedding_model="text-embedding-3-large",
            prompt_tokens=0,
            completion_tokens=0,
            documents_retrieved=all_sources,
            extra_llm_calls=None
        )

        return answer, all_sources, cost_log

    # Final answer over the merged set
    answer, token_usage = process_query_with_cost_tracking(user_question, all_enhanced_docs, llm_obj)


    cost_log = interaction_logger.log_interaction(
        search_mode="Leonardo",
        question=user_question,
        answer=answer,
        chat_model="gpt-3.5-turbo-16k",
        embedding_model="text-embedding-3-large",
        prompt_tokens=token_usage.get('prompt_tokens', 0),
        completion_tokens=token_usage.get('completion_tokens', 0),
        documents_retrieved=all_sources,
        extra_llm_calls=None
    )

    return answer, all_sources, cost_log




def handle_donatello_search_with_costs(user_question: str, doc_name: str, searcher, metadata_attrs) -> tuple:
    """Enhanced Donatello with cost tracking - includes extra LLM filtering cost"""
    embeddings_results, json_bm25_results = run_both_searches_concurrent(
        user_question=user_question,
        doc_name=doc_name
    )
    
    bm25_results = dic_to_doc(json_bm25_results)
    
    embeddings_results = enhance_embeddings_with_hierarchy_dynamic(
        embeddings_results, searcher, metadata_attrs
    )
    
    all_doc = interleave_search_results(embeddings_results, bm25_results)
    
    enhanced_docs = enhance_documents_dynamic(all_doc, metadata_attrs)
    
    # Filter documents with cost tracking (NEW!)
    filtered_docs, filter_token_usage = filter_documents_by_relevance_with_cost_tracking(
        question=user_question,
        documents_input=enhanced_docs,
        llm=llm
    )
    
    # Generate answer
    answer, answer_token_usage = process_query_with_cost_tracking(user_question, filtered_docs)
    sources = filtered_docs
    
    # Log with extra LLM costs
    cost_log = interaction_logger.log_interaction(
        search_mode="Donatello",
        question=user_question,
        answer=answer,
        chat_model="gpt-3.5-turbo-16k",
        embedding_model="text-embedding-3-large",
        prompt_tokens=answer_token_usage.get('prompt_tokens', 0),
        completion_tokens=answer_token_usage.get('completion_tokens', 0),
        documents_retrieved=sources,
        extra_llm_calls={
            'model': 'gpt-3.5-turbo-16k',
            'prompt_tokens': filter_token_usage.get('prompt_tokens', 0),
            'completion_tokens': filter_token_usage.get('completion_tokens', 0)
        }
    )
    
    return answer, sources, cost_log



def convert_numpy_to_native(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    Handles nested dictionaries, lists, and various NumPy data types.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def filter_documents_by_relevance_with_cost_tracking(
    question: str,
    documents_input,
    llm
) -> Tuple[List[Document], Dict]:
    """
    Enhanced version that returns both filtered documents AND token usage
    
    Returns:
        Tuple of (filtered_documents, token_usage_dict)
    """
    
    # Handle list of lists (flatten if needed)
    if isinstance(documents_input, list) and len(documents_input) > 0:
        if isinstance(documents_input[0], list):
            documents = []
            for doc_list in documents_input:
                documents.extend(doc_list)
        else:
            documents = documents_input
    else:
        documents = []
    
    # Format for LLM
    formatted_chunks = ""
    for i, doc in enumerate(documents, 1):
        formatted_chunks += f"\n=== CHUNK {i} ===\n"
        formatted_chunks += f"Metadata: {json.dumps(convert_numpy_to_native(doc.metadata), indent=2)}\n"
        formatted_chunks += f"Content: {doc.page_content}\n"
    
    donatello_prompt_template = f"""You are an expert document analyst. Analyze document chunks and determine relevance to a question.

    TASK:
    1. Evaluate each chunk's relevance to the question, make sure to pay attention to the metadata when ranking. Make sure the answers come from relevant chapter, sections, subchapters, etc...
    2. Score relevance from 0-100 (100 = perfect match for answering the question)
    3. Only return chunks of text that are relevant to answer the question. 

    Question: "{question}"

    Document Chunks:
    {formatted_chunks}

    Return JSON: {{"relevant_chunks": [{{"chunk_number": 1, "relevance_score": 95}}]}}"""
    
    # Count tokens in the prompt
    prompt_tokens = cost_tracker.count_tokens(donatello_prompt_template)
    
    try:
        response = llm.invoke(donatello_prompt_template)
        response_text = response.content.strip()
        
        # Count completion tokens
        completion_tokens = cost_tracker.count_tokens(response_text)
        
        # Clean JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Convert back to Document objects
        relevant_docs = []
        for chunk_info in result.get('relevant_chunks', []):
            chunk_number = chunk_info.get('chunk_number', 1)
            relevance_score = chunk_info.get('relevance_score', 0)
            
            # Get the original document (chunk_number is 1-indexed)
            if 1 <= chunk_number <= len(documents):
                original_doc = documents[chunk_number - 1]
                
                # Create new Document with relevance info added to metadata
                new_metadata = original_doc.metadata.copy()
                new_metadata['relevance_score'] = relevance_score
                
                relevant_doc = Document(
                    page_content=original_doc.page_content,
                    metadata=new_metadata
                )
                relevant_docs.append(relevant_doc)
        
        # Sort by relevance score (highest first)
        relevant_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        # Return both documents and token usage
        token_usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'model': 'gpt-3.5-turbo-16k'  # Same as main model
        }
        
        return relevant_docs, token_usage
        
    except Exception as e:
        print(f"Error filtering documents: {e}")
        # Return empty list and zero tokens on error
        return [], {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


#@st.cache_resource
#def display_pdf_in_popover(pdf_path, page_number, button_text):
#    with st.popover(button_text):
#        try:
#            with open(pdf_path, "rb") as f:
#                pdf_data = f.read()
#            
#            pdf_base64 = base64.b64encode(pdf_data).decode()
#            
#            # Display PDF in popover
#            st.markdown(f"""
#            <iframe src="data:application/pdf;base64,{pdf_base64}#page={page_number}" 
#                    width="600" height="800" style="border: none;">
#            </iframe>
#            """, unsafe_allow_html=True)
#            
#        except FileNotFoundError:
#            st.error(f"PDF not found: {pdf_path}")


# UI with performance improvements
st.header("⚖️ Legal Juris", divider="orange")
st.markdown("*Your AI-powered legal research assistant for Texas law*")

# Sidebar
with st.sidebar:
    st.title("📜 Select Document")
    
    multi_select_enabled = st.checkbox(
            "Enable multi-select",
            value=st.session_state.multi_select_mode
        )
    
    if multi_select_enabled != st.session_state.multi_select_mode:
        previous_mode = st.session_state.multi_select_mode
        st.session_state.multi_select_mode = multi_select_enabled

        if not multi_select_enabled:
            if st.session_state.loaded_docs:
                st.session_state.current_doc = st.session_state.loaded_docs[0]
            elif st.session_state.selected_docs:
                st.session_state.current_doc = st.session_state.selected_docs[0]
            else:
                st.session_state.current_doc = None
        
        for key in ['current_question', 'current_answer', 'current_sources',
            'current_doc_name', 'current_search_mode', 'current_record_id']:
            if key in st.session_state:
                del st.session_state[key]

        for key in list(st.session_state.keys()):
            if key.startswith('feedback_') or key.startswith('survey_'):
                del st.session_state[key]

    categories = {
    "Constitution": [
        "Texas Constitution",
        "United States Constitution"
    ],

    "Texas Codes & Laws": [
        "Auxiliary Water Laws",
        "Texas Agriculture Code",
        "Texas Alcoholic Beverage Code",
        "Texas Business and Commerce Code",
        "Texas Business Organizations Code",
        "Texas Civil Practice And Remedies Code",
        "Texas Code of Criminal Procedure",
        "Texas Education Code",
        "Texas Election Code",
        "Texas Estates Code",
        "Texas Family Code",
        "Texas Finance Code",
        #"Texas Government Code",
        "Texas Human Resources Code",
        "Texas Insurance Code",
        "Texas Labor Code",
        "Texas Local Government Code",
        "Texas Natural Resources Code",
        "Texas Occupations Code",
        "Texas Parks And Wild Life Code",
        "Texas Penal Code",
        "Texas Property Code",
        "Texas Tax Code",
        "Texas Transportation Code",
        "Texas Utilities Code",
        "Texas Water Code"
    ],
    "Texas Rules": [
        "Education Rules on Guardianship, Alternatives to Guardianship, and Supports and Services for Proposed Wards and Wards",
        "Judicial Branch Certification Commission Rules",
        "Judicial Bypass Rules under Ch. 33 of the Family Code",
        "Procedural Rules for the State Commission on Judicial Conduct",
        "Rules for Magistrates in Inmate Litigation and Litigation Involving Certain Civilly Committed Individuals",
        "Rules of Judicial Education",
        "Statewide Rules Governing Electronic Filing in Criminal Cases",
        "Texas Code of Judicial Conduct",
        "Texas Disciplinary Rules of Professional Conduct",
        "Texas Rules of Appellate Procedure",
        "Texas Rules of Civil Procedure",
        "Texas Rules of Disciplinary Procedure",
        "Texas Rules of Evidence",
        "Texas Rules of Judicial Administration"
    ],

    "Texas Standards, Guidelines, and Handbooks": [
        "Ethical Guidelines for Mediators",
        "JCIT Technology Standards",
        "Texas Administrative Law Handbook",
        "Uniform Format Manual for Texas Reporters Records"
    ],
    "Federal Rules": [
        "Federal Rules of Appellate Procedure",
        "Federal Rules of Bankruptcy Procedure",
        "Federal Rules of Civil Procedure",
        "Federal Rules of Criminal Procedure",
        "Federal Rules of Evidence"
    ]
    }
    
    all_documents = []
    doc_to_category = {}

    for category, docs in categories.items():
        for doc in docs:
            all_documents.append(doc)
            doc_to_category[doc] = category


   
    doc_name = None
    search_mode = None
    if not st.session_state.multi_select_mode:
        selected_category = st.selectbox(
            "Category",
            options=["Select a category..."] + list(categories.keys()),
            index=0,
            help="Choose a legal document category",
            key="single_select_category"
        )

        if selected_category and selected_category != "Select a category...":
            st.write("**Choose a document:**")
            doc_name = st.radio(
                "Choose a document:",
                options=categories[selected_category],
                index=(
                    categories[selected_category].index(st.session_state.current_doc)
                    if st.session_state.current_doc in categories[selected_category]
                    else None
                ),
                help="Select the specific document to search",
                label_visibility="collapsed",
                key="single_select_doc_radio"
            )
        else:
            doc_name = None
        
        if doc_name:
            if doc_name != st.session_state.get('current_doc'):
                st.session_state.current_doc = doc_name

                for key in ['current_question', 'current_answer', 'current_sources',
                            'current_doc_name', 'current_search_mode', 'current_record_id']:
                    if key in st.session_state:
                        del st.session_state[key]

                for key in list(st.session_state.keys()):
                    if key.startswith('feedback_') or key.startswith('survey_'):
                        del st.session_state[key]

            doc_resources = initialize_document_resources(doc_name)

            if doc_resources:
                metadata_attrs = doc_resources['metadata_attrs']
                knowledge_base = doc_resources['knowledge_base']
                llm = doc_resources['llm']
                retriever = doc_resources['retriever']
                st.session_state.loaded_docs = [doc_name]
                st.session_state.multi_doc_resources = {}
                kb_status = "✅ Loaded"
                st.info(f"📊 Knowledge: {kb_status}")
                search_mode = "Leonardo"
                bm25_name = doc_name.replace(" ", "_") + "_bm25.pkl"
                searcher = LegalBM25Search(f"bm25//{bm25_name}")
            else:
                st.error("Failed to load document resources")
    
    else:
        st.write("**Choose documents:**")

        new_selected_docs = []

        for category, docs in categories.items():
            with st.expander(category, expanded=False):
                for doc in docs:
                    checked = st.checkbox(
                            doc,
                            value=doc in st.session_state.selected_docs,
                            key=f"multi_doc_{doc}"
                    )
                    if checked:
                        new_selected_docs.append(doc)

        st.session_state.selected_docs = new_selected_docs

        load_documents_clicked = st.button(
            "📚 Load Documents",
            use_container_width=True,
            key="multi_load_documents"
        )
        if load_documents_clicked:
            if st.session_state.selected_docs:
                loaded_resources = {}
                failed_docs = []

                for selected_doc in st.session_state.selected_docs:
                    doc_resources = initialize_document_resources(selected_doc)

                    if doc_resources:
                        bm25_name = selected_doc.replace(" ", "_") + "_bm25.pkl"
                        loaded_resources[selected_doc] = {
                            "metadata_attrs": doc_resources["metadata_attrs"],
                            "knowledge_base": doc_resources["knowledge_base"],
                            "llm": doc_resources["llm"],
                            "retriever": doc_resources["retriever"],
                            "searcher": LegalBM25Search(f"bm25//{bm25_name}"),
                            "search_mode": "Leonardo"
                        }
                    else:
                        failed_docs.append(selected_doc)

                if loaded_resources:
                    st.session_state.multi_doc_resources = loaded_resources
                    st.session_state.loaded_docs = list(loaded_resources.keys())
                    st.session_state.current_doc = st.session_state.loaded_docs[0]

                    st.success(f"Loaded {len(st.session_state.loaded_docs)} document(s)")
                    st.write("Loaded docs:", st.session_state.loaded_docs)
                    st.write("Resource keys:", list(st.session_state.multi_doc_resources.keys()))
                    st.write("Resource count:", len(st.session_state.multi_doc_resources))


                    if failed_docs:
                        st.warning("Failed to load: " + ", ".join(failed_docs))
                else:
                    st.session_state.multi_doc_resources = {}
                    st.session_state.loaded_docs = []
                    st.session_state.current_doc = None
                    st.error("Failed to load selected documents.")
            else:
                st.warning("Please select at least one document.")

        if st.session_state.loaded_docs:
            st.info(f"📊 Knowledge: ✅ {len(st.session_state.loaded_docs)} documents loaded")



        
  
# CHANGED: Single column layout instead of two columns
st.subheader("Ask Juris")

# Question input
user_question = st.text_area(
    "Your legal question:",
    placeholder="e.g., What are the grounds for divorce in Texas?",
    height=30,
    help="Ask specific questions about the selected document"
)

# Buttons
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)


def enhance_documents_dynamic(alldoc, metadata_attrs):
    """Add metadata to page_content using existing hierarchical order"""
    enhanced_docs = []
    
    # Use the already-ordered attribute names
    attr_names = [attr.name for attr in metadata_attrs]
    
    for doc in alldoc:
        # Create metadata header in hierarchical order
        metadata_parts = []
        
        for attr_name in attr_names:
            if attr_name in doc.metadata and doc.metadata[attr_name]:
                value = doc.metadata[attr_name]
                display_name = attr_name.replace('_', ' ').title()
                metadata_parts.append(f"{display_name}: {value}")
        
        # Enhanced content with metadata header
        if metadata_parts:
            metadata_header = "[" + " | ".join(metadata_parts) + "]"
            enhanced_content = f"{metadata_header}\n{doc.page_content}"
        else:
            enhanced_content = doc.page_content
        
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata=doc.metadata
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs


if clear_clicked:
    for key in [
        'chat_history',
        'current_question',
        'current_answer',
        'current_sources',
        'current_doc_name',
        'current_search_mode',
        'current_record_id',
        'search_start_time',
        'current_response_time'
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
# In your main search section, replace the search button logic:

if search_clicked:
    if not user_question:
        st.warning("Please enter a question")
    elif st.session_state.multi_select_mode and not st.session_state.loaded_docs:
        st.warning("Please load at least one document first")
    elif not st.session_state.multi_select_mode and not doc_name:
        st.warning("Please select a document first")
    else:
        # Clear any previous feedback state for new search
        for key in list(st.session_state.keys()):
            if key.startswith('feedback_') or key.startswith('survey_'):
                del st.session_state[key]
        
        # START TIMING: Record when search begins (question submitted)
        start_time = time.time()
        st.session_state['search_start_time'] = start_time
        
        with st.spinner("🔍 Searching through legal documents..."):
            if st.session_state.multi_select_mode:
                answer, sources, cost_data = handle_multi_doc_leonardo_search_with_costs(
                    user_question,
                    st.session_state.multi_doc_resources
                )
                active_doc_label = ", ".join(st.session_state.loaded_docs)
                active_search_mode = "Leonardo"
            else:
                answer, sources, cost_data = handle_leonardo_search_with_costs(
                        user_question, doc_name, retriever, searcher, metadata_attrs, llm
                    )
                active_doc_label = doc_name
                active_search_mode = search_mode

        # Store NEW current Q&A in session state (without response time yet)
        st.session_state['current_question'] = user_question
        st.session_state['current_answer'] = answer
        st.session_state['current_sources'] = sources
        st.session_state['current_doc_name'] = active_doc_label
        st.session_state['current_search_mode'] = active_search_mode

        
# Always show current question/answer if it exists
if 'current_question' in st.session_state and 'current_answer' in st.session_state:
    st.warning("⚠️ Always verify answers!")
    
    st.subheader("📋 Answer")
        # MEASURE COMPLETE RESPONSE TIME HERE (after answer is displayed)

    search_type = f"🧠 {st.session_state.get('current_search_mode', 'Unknown')}"
    st.markdown(
        f"**Document:** {st.session_state.get('current_doc_name', 'Unknown')} | "
        f"**Mode:** {search_type}"
    )
    # Display the answer
    render_plain_wrapped_text(st.session_state['current_answer'])
    #st.write(st.session_state['current_sources'])

    # Display sources
    if 'current_sources' in st.session_state:
        display_sources_with_tabs(st.session_state['current_sources'], st.session_state['current_doc_name'])

    if 'search_start_time' in st.session_state and 'current_response_time' not in st.session_state:
        display_time = time.time()
        total_response_time = round(display_time - st.session_state['search_start_time'], 2)
        st.session_state['current_response_time'] = total_response_time
        
        # Log to Airtable initially without response time (will be updated after display)
        #with st.spinner("📝 Logging interaction..."):
        success, record_id = log_to_airtable_with_costs(
                question=user_question,
                answer=answer,
                doc_name=active_doc_label,
                search_mode=active_search_mode,
                response_time=total_response_time,
                source_count=len(sources),
                cost_data=cost_data
            )

        if success:
            st.session_state['current_record_id'] = record_id

        del st.session_state['search_start_time']
        del st.session_state['current_response_time']
    
    
    # Display feedback buttons
    record_id = st.session_state.get('current_record_id')
    if record_id:
        st.markdown("---")
        st.markdown("**Was this answer helpful?**")
        display_feedback_buttons("current_question", record_id)

# Check if we need to show feedback survey for current question
if 'current_record_id' in st.session_state:
    question_id = "current_question"
    feedback_status = st.session_state.get(f'feedback_{question_id}', None)
    survey_submitted = st.session_state.get(f'survey_submitted_{question_id}', False)
    
    # Show survey if thumbs down was clicked but survey not submitted
    if feedback_status == 'thumbs_down' and not survey_submitted:
        display_feedback_survey(question_id)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
💡 <b>Performance Tips:</b> Use specific questions • Select relevant documents<br>
""", unsafe_allow_html=True)

