import google.generativeai as genai
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GeminiLLMProvider:
    """Google Gemini 2.5 Flash provider for cloud deployment"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Safety settings for RAG use case
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Rate limiting for free tier (15 RPM, 1M tokens/month)
        self.rate_limiter = GeminiRateLimiter()
    
    def generate_rag_response(self, query: str, context: str, conversation_history: List[Dict] = None) -> Dict:
        """Generate RAG response using Gemini with context and conversation history"""
        
        if not self.rate_limiter.can_make_request():
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "fallback_suggested": True
            }
        
        try:
            prompt = self._build_rag_prompt(query, context, conversation_history)
            
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Record request for rate limiting
            token_count = getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            self.rate_limiter.record_request(token_count)
            
            return {
                "success": True,
                "response": response.text,
                "token_count": token_count,
                "provider": "gemini-2.0-flash"
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                "success": False,
                "error": f"Gemini API error: {str(e)}",
                "fallback_suggested": True
            }
    
    def _build_rag_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Build RAG prompt with context and conversation history"""
        
        prompt_parts = [
            "You are a helpful AI assistant with access to a knowledge base.",
            "Provide educational, accurate responses based on the context provided.",
            ""
        ]
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Conversation History:")
            for entry in conversation_history[-5:]:  # Keep last 5 exchanges
                prompt_parts.append(f"User: {entry.get('user', '')}")
                prompt_parts.append(f"Assistant: {entry.get('assistant', '')}")
            prompt_parts.append("")
        
        # Add relevant context
        prompt_parts.extend([
            "Relevant Context from Knowledge Base:",
            context,
            "",
            f"User Question: {query}",
            "",
            "Please provide a helpful response based on the context provided. If the context doesn't contain relevant information, say so clearly."
        ])
        
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if Gemini API is available and within rate limits"""
        return bool(self.api_key) and self.rate_limiter.can_make_request()
    
    def get_cost_per_token(self) -> float:
        """Get cost per token for Gemini (free tier: $0, paid: varies)"""
        return 0.0  # Free tier


class GeminiRateLimiter:
    """Rate limiter for Gemini free tier (15 RPM, 1M tokens/month)"""
    
    def __init__(self):
        from datetime import datetime, timedelta
        self.free_tier_limits = {
            'requests_per_minute': 15,
            'tokens_per_month': 1000000,
            'requests_per_day': 1500
        }
        self.request_history = []
        self.token_history = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limits"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # Clean old requests (older than 1 minute)
        cutoff_minute = now - timedelta(minutes=1)
        self.request_history = [req for req in self.request_history if req > cutoff_minute]
        
        # Check RPM limit
        return len(self.request_history) < self.free_tier_limits['requests_per_minute']
    
    def record_request(self, tokens_used: int):
        """Record a request for rate limiting purposes"""
        from datetime import datetime
        
        now = datetime.now()
        self.request_history.append(now)
        self.token_history.append((now, tokens_used))
        
        # Log usage for monitoring
        logger.info(f"Gemini request recorded: {tokens_used} tokens, {len(self.request_history)} requests in last minute") 