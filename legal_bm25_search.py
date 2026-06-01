import pickle
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any
import re

def ensure_nltk_data():
    """Download NLTK data only if not already present."""
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"📥 Downloading {resource_name}...")
            nltk.download(resource_name, quiet=True)

ensure_nltk_data()

class LegalBM25Search:
    """
    Legal document BM25 search system that loads pre-built index.
    """
    
    def __init__(self, index_path: str):
        """Initialize by loading the saved BM25 index."""
        self.index_path = index_path
        self.bm25 = None
        self.doc_mapping = None
        self.index_metadata = None
        self.hierarchy_levels = None
        self.load_index()
    
    def load_index(self):
        """Load the BM25 index from pickle file."""
        #print(f"Loading legal BM25 index from {self.index_path}...")
        
        try:
            with open(self.index_path, "rb") as f:
                index_package = pickle.load(f)
            
            self.bm25 = index_package['bm25_index']
            self.doc_mapping = index_package['doc_mapping']
            self.index_metadata = index_package['index_metadata']
            self.hierarchy_levels = index_package['hierarchy_levels']
            
            #print(f"✅ Index loaded successfully!")
            #print(f"📊 Contains {self.index_metadata['total_documents']} legal documents")
            #print(f"📚 Sources: {', '.join(self.index_metadata['sources'])}")
            #print(f"🏗️ Hierarchy: {' > '.join(self.hierarchy_levels)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"BM25 index not found at {self.index_path}")
        except Exception as e:
            raise Exception(f"Error loading BM25 index: {str(e)}")
    
    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess search query using same method as indexing.
        Must match the preprocessing used during index creation.
        """
        # Same preprocessing as used during indexing
        query = query.lower()
        query = re.sub(r'\s+', ' ', query).strip()
        tokens = word_tokenize(query)
        tokens = [token for token in tokens if token.isalnum()]
        return tokens

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform BM25 search and return ranked results.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing document info and scores
        """
        # Preprocess the query
        query_tokens = self.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        #print(f"🔍 Searching for: '{query}'")
        #print(f"📝 Query tokens: {query_tokens}")
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k document indices
        top_indices = scores.argsort()[-top_k:][::-1]  # Sort descending
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc_info = self.doc_mapping[idx].copy()
                doc_info['bm25_score'] = float(scores[idx])
                doc_info['rank'] = len(results) + 1
                results.append(doc_info)
        
        #print(f"📋 Found {len(results)} relevant documents")
        return results

    def find_by_any_level(self, search_term: str, level: str, exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Find documents by any hierarchy level with flexible level specification.
        
        Args:
            search_term: Term to search for
            level: Hierarchy level to search ('section', 'chapter', 'title', 'subtitle', etc.)
            exact_match: If True, requires exact match; if False, uses substring matching
            
        Returns:
            List of matching documents
        """
        if level not in self.hierarchy_levels:
            available_levels = ", ".join(self.hierarchy_levels)
            print(f"⚠️ Warning: '{level}' not in available hierarchy levels.")
            print(f"📋 Available levels: {available_levels}")
            return []
        
        matching_docs = []
        
        for doc_info in self.doc_mapping:
            level_value = doc_info['metadata'].get(level, '')
            
            if not level_value:  # Skip if this document doesn't have this level
                continue
                
            if exact_match:
                if level_value.lower() == search_term.lower():
                    matching_docs.append(doc_info.copy())
            else:
                if search_term.lower() in level_value.lower():
                    matching_docs.append(doc_info.copy())
        
        print(f"🔍 Found {len(matching_docs)} documents in '{level}' containing '{search_term}'")
        return matching_docs

    def find_by_multiple_levels(self, search_criteria: Dict[str, str], match_all: bool = False) -> List[Dict[str, Any]]:
        """
        Find documents matching multiple hierarchy levels.
        
        Args:
            search_criteria: Dict like {'chapter': 'PARENT-CHILD', 'section': 'RIGHTS'}
            match_all: If True, document must match ALL criteria; if False, match ANY
            
        Returns:
            List of matching documents
        """
        matching_docs = []
        
        for doc_info in self.doc_mapping:
            matches = []
            
            for level, search_term in search_criteria.items():
                if level not in self.hierarchy_levels:
                    print(f"⚠️ Skipping unknown level: {level}")
                    continue
                    
                level_value = doc_info['metadata'].get(level, '').lower()
                if search_term.lower() in level_value:
                    matches.append(True)
                else:
                    matches.append(False)
            
            # Apply matching logic
            if match_all and all(matches):
                matching_docs.append(doc_info.copy())
            elif not match_all and any(matches):
                matching_docs.append(doc_info.copy())
        
        return matching_docs