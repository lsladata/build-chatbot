import tiktoken
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class AzureOpenAICostTracker:
    """
    Comprehensive cost tracking for Azure OpenAI API calls
    Supports embeddings and chat completion cost calculations
    """
    
    def __init__(self):
        # Azure OpenAI pricing per 1M tokens (as of January 2024)
        # Update these if Microsoft changes pricing
        self.pricing = {
            # Chat models
            "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},  # per 1M tokens
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            
            # Embedding models (per 1K tokens, converted to per 1M for consistency)
            "text-embedding-3-large": {"input": 130.0, "output": 0},  # $0.13 per 1K = $130 per 1M
            "text-embedding-3-small": {"input": 20.0, "output": 0},   # $0.02 per 1K = $20 per 1M
            "text-embedding-ada-002": {"input": 100.0, "output": 0}   # $0.10 per 1K = $100 per 1M
        }
        
        # Initialize tokenizer for GPT-3.5/4 models
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Warning: Failed to initialize tiktoken: {e}")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not self.encoding:
            # Fallback: estimate 4 characters per token
            return len(text) // 4
        
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"Token counting error: {e}")
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in a list of chat messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name for token counting adjustments
            
        Returns:
            Total token count for the messages
        """
        if not self.encoding:
            # Fallback estimation
            total_text = " ".join([msg.get("content", "") for msg in messages])
            return len(total_text) // 4
        
        # Tokens per message and name (varies by model)
        tokens_per_message = 3
        tokens_per_name = 1
        
        if model in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", 
                     "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"]:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        
        # Every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3
        return num_tokens
    
    def calculate_chat_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """
        Calculate cost for a chat completion API call
        
        Args:
            model: Model name (e.g., 'gpt-3.5-turbo-16k')
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Dict with input_cost, output_cost, and total_cost
        """
        # Normalize model name
        model_key = self._normalize_model_name(model)
        
        if model_key not in self.pricing:
            print(f"Warning: Unknown model '{model}', using default pricing")
            model_key = "gpt-3.5-turbo-16k"
        
        pricing = self.pricing[model_key]
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
        output_cost = (completion_tokens * pricing["output"]) / 1_000_000
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    def calculate_embedding_cost(self, model: str, total_tokens: int) -> Dict[str, float]:
        """
        Calculate cost for an embedding API call
        
        Args:
            model: Embedding model name (e.g., 'text-embedding-3-large')
            total_tokens: Total number of tokens embedded
            
        Returns:
            Dict with total_cost and token count
        """
        model_key = self._normalize_model_name(model)
        
        if model_key not in self.pricing:
            print(f"Warning: Unknown embedding model '{model}', using default pricing")
            model_key = "text-embedding-3-large"
        
        pricing = self.pricing[model_key]
        
        # Embeddings only have input cost (pricing already converted to per 1M)
        total_cost = (total_tokens * pricing["input"]) / 1_000_000
        
        return {
            "total_cost": round(total_cost, 6),
            "model": model,
            "total_tokens": total_tokens
        }
    
    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model names for consistent pricing lookup
        
        Args:
            model: Raw model name from API
            
        Returns:
            Normalized model name
        """
        model_lower = model.lower()
        
        # Chat models
        if "gpt-3.5-turbo-16k" in model_lower or "chat3-5-16" in model_lower:
            return "gpt-3.5-turbo-16k"
        elif "gpt-3.5-turbo" in model_lower:
            return "gpt-3.5-turbo"
        elif "gpt-4-turbo" in model_lower:
            return "gpt-4-turbo"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        
        # Embedding models
        elif "text-embedding-3-large" in model_lower or "model3large" in model_lower:
            return "text-embedding-3-large"
        elif "text-embedding-3-small" in model_lower:
            return "text-embedding-3-small"
        elif "ada" in model_lower or "ada-002" in model_lower:
            return "text-embedding-ada-002"
        
        return model
    
    def estimate_document_tokens(self, documents: List[Any]) -> int:
        """
        Estimate total tokens in a list of documents
        
        Args:
            documents: List of Document objects with page_content attribute
            
        Returns:
            Total token count estimate
        """
        total_tokens = 0
        for doc in documents:
            if hasattr(doc, 'page_content'):
                total_tokens += self.count_tokens(doc.page_content)
            elif isinstance(doc, dict) and 'content' in doc:
                total_tokens += self.count_tokens(doc['content'])
            else:
                total_tokens += self.count_tokens(str(doc))
        
        return total_tokens


class InteractionCostLogger:
    """
    Logger for tracking costs of complete question-answer interactions
    Handles all 4 ninja turtle search modes
    """
    
    def __init__(self, cost_tracker: AzureOpenAICostTracker):
        self.cost_tracker = cost_tracker
    
    def log_interaction(
        self,
        search_mode: str,
        question: str,
        answer: str,
        chat_model: str,
        embedding_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        documents_retrieved: List[Any],
        extra_llm_calls: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a complete interaction with cost breakdown
        
        Args:
            search_mode: One of ['Michelangelo', 'Raphael', 'Leonardo', 'Donatello']
            question: User's question
            answer: AI's answer
            chat_model: Chat model used (e.g., 'gpt-3.5-turbo-16k')
            embedding_model: Embedding model used (e.g., 'text-embedding-3-large')
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            documents_retrieved: List of retrieved documents
            extra_llm_calls: Dict with info about additional LLM calls (for Donatello)
            
        Returns:
            Complete cost breakdown dictionary
        """
        # Calculate chat completion cost
        chat_cost = self.cost_tracker.calculate_chat_cost(
            model=chat_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        # Estimate embedding cost from retrieved documents
        doc_tokens = self.cost_tracker.estimate_document_tokens(documents_retrieved)
        embedding_cost = self.cost_tracker.calculate_embedding_cost(
            model=embedding_model,
            total_tokens=doc_tokens
        )
        
        # Calculate extra costs for Donatello's document filtering
        extra_cost = {"total_cost": 0, "prompt_tokens": 0, "completion_tokens": 0}
        if extra_llm_calls and search_mode == "Donatello":
            extra_cost = self.cost_tracker.calculate_chat_cost(
                model=extra_llm_calls.get("model", chat_model),
                prompt_tokens=extra_llm_calls.get("prompt_tokens", 0),
                completion_tokens=extra_llm_calls.get("completion_tokens", 0)
            )
        
        # Calculate total cost
        total_cost = (
            chat_cost["total_cost"] + 
            embedding_cost["total_cost"] + 
            extra_cost["total_cost"]
        )
        
        # Create comprehensive log entry
        cost_log = {
            "timestamp": datetime.now().isoformat(),
            "search_mode": search_mode,
            "question": question[:200],  # Truncate for logging
            "answer": answer[:200],      # Truncate for logging
            
            # Chat completion costs
            "chat_model": chat_model,
            "chat_prompt_tokens": prompt_tokens,
            "chat_completion_tokens": completion_tokens,
            "chat_total_tokens": prompt_tokens + completion_tokens,
            "chat_input_cost": chat_cost["input_cost"],
            "chat_output_cost": chat_cost["output_cost"],
            "chat_total_cost": chat_cost["total_cost"],
            
            # Embedding costs
            "embedding_model": embedding_model,
            "embedding_tokens": doc_tokens,
            "embedding_cost": embedding_cost["total_cost"],
            "documents_count": len(documents_retrieved),
            
            # Extra LLM costs (Donatello)
            "extra_llm_cost": extra_cost["total_cost"],
            "extra_llm_prompt_tokens": extra_cost.get("prompt_tokens", 0),
            "extra_llm_completion_tokens": extra_cost.get("completion_tokens", 0),
            
            # Total cost
            "total_cost": round(total_cost, 6),
            
            # Cost breakdown percentage
            "cost_breakdown": {
                "chat_percentage": round((chat_cost["total_cost"] / total_cost * 100) if total_cost > 0 else 0, 2),
                "embedding_percentage": round((embedding_cost["total_cost"] / total_cost * 100) if total_cost > 0 else 0, 2),
                "extra_percentage": round((extra_cost["total_cost"] / total_cost * 100) if total_cost > 0 else 0, 2)
            }
        }
        
        return cost_log
    
    def format_for_airtable(self, cost_log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format cost log for Airtable insertion
        
        Args:
            cost_log: Cost log dictionary from log_interaction
            
        Returns:
            Dictionary formatted for Airtable API
        """
        return {
            "Timestamp": cost_log["timestamp"],
            "Search Mode": cost_log["search_mode"],
            "Question": cost_log["question"],
            "Answer": cost_log["answer"],
            
            # Chat costs
            "Chat Model": cost_log["chat_model"],
            "Chat Prompt Tokens": cost_log["chat_prompt_tokens"],
            "Chat Completion Tokens": cost_log["chat_completion_tokens"],
            "Chat Total Tokens": cost_log["chat_total_tokens"],
            "Chat Input Cost": cost_log["chat_input_cost"],
            "Chat Output Cost": cost_log["chat_output_cost"],
            "Chat Total Cost": cost_log["chat_total_cost"],
            
            # Embedding costs
            "Embedding Model": cost_log["embedding_model"],
            "Embedding Tokens": cost_log["embedding_tokens"],
            "Embedding Cost": cost_log["embedding_cost"],
            "Documents Retrieved": cost_log["documents_count"],
            
            # Extra costs (Donatello)
            "Extra LLM Cost": cost_log["extra_llm_cost"],
            "Extra LLM Prompt Tokens": cost_log["extra_llm_prompt_tokens"],
            "Extra LLM Completion Tokens": cost_log["extra_llm_completion_tokens"],
            
            # Total
            "Total Cost": cost_log["total_cost"],
            
            # Percentages (stored as text for easier reading)
            "Chat Cost Percentage": f"{cost_log['cost_breakdown']['chat_percentage']}%",
            "Embedding Cost Percentage": f"{cost_log['cost_breakdown']['embedding_percentage']}%",
            "Extra Cost Percentage": f"{cost_log['cost_breakdown']['extra_percentage']}%"
        }