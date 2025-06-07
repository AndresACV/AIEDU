"""
Smart LLM Router for AIEDU RAG System
Intelligently routes requests between Ollama (local) and Gemini (cloud) providers
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class ProviderPreference(Enum):
    PRIVACY = "privacy"        # Prefer local (Ollama)
    PERFORMANCE = "performance" # Prefer cloud (Gemini)
    AUTO = "auto"             # Smart selection
    COST_CONSCIOUS = "cost"   # Minimize API costs

@dataclass
class ProviderPerformance:
    """Track provider performance metrics"""
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    total_requests: int = 0
    total_tokens: int = 0

class SmartLLMRouter:
    """Smart router that selects optimal LLM provider based on multiple factors"""
    
    def __init__(self, user_preference: ProviderPreference = ProviderPreference.AUTO):
        self.user_preference = user_preference
        self.performance_history = {
            'ollama': ProviderPerformance(),
            'gemini': ProviderPerformance()
        }
        
        # Circuit breaker settings
        self.failure_threshold = 3
        self.recovery_timeout = 300  # 5 minutes
        
        # Cost tracking
        self.daily_cost_limit = 1.0  # $1 per day for free tier safety
        self.current_daily_cost = 0.0
        self.cost_reset_date = datetime.now().date()
        
        logger.info(f"SmartLLMRouter initialized with preference: {user_preference.value}")
    
    def select_provider(self, query: str, context: str = "") -> str:
        """Select the optimal provider for this request"""
        
        # Reset daily cost if new day
        self._reset_daily_cost_if_needed()
        
        # Get available providers
        available_providers = self._get_available_providers()
        
        if not available_providers:
            raise Exception("No LLM providers available")
        
        # Apply user preference logic
        if self.user_preference == ProviderPreference.PRIVACY:
            return self._prefer_local(available_providers)
        
        elif self.user_preference == ProviderPreference.PERFORMANCE:
            return self._prefer_cloud(available_providers)
        
        elif self.user_preference == ProviderPreference.COST_CONSCIOUS:
            return self._prefer_cost_effective(available_providers)
        
        else:  # AUTO mode
            return self._smart_selection(available_providers, query, context)
    
    def execute_request(self, provider_name: str, query: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Execute request with the selected provider and track performance"""
        
        start_time = time.time()
        
        try:
            if provider_name == 'ollama':
                result = self._execute_ollama_request(query, context, conversation_history)
            elif provider_name == 'gemini':
                result = self._execute_gemini_request(query, context, conversation_history)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            # Track successful request
            response_time = time.time() - start_time
            self._track_success(provider_name, response_time, result.get('token_count', 0))
            
            result['provider_used'] = provider_name
            result['response_time'] = response_time
            
            logger.info(f"✅ {provider_name} request successful: {response_time:.2f}s, {result.get('token_count', 0)} tokens")
            
            return result
            
        except Exception as e:
            # Track failed request
            response_time = time.time() - start_time
            self._track_failure(provider_name, str(e))
            
            logger.error(f"❌ {provider_name} request failed: {str(e)}")
            
            # Try fallback provider
            return self._try_fallback(provider_name, query, context, conversation_history, str(e))
    
    def _get_available_providers(self) -> List[str]:
        """Get list of currently available providers"""
        available = []
        
        # Check Ollama availability
        if self._is_ollama_available():
            available.append('ollama')
        
        # Check Gemini availability (including cost limits and circuit breaker)
        if self._is_gemini_available():
            available.append('gemini')
        
        return available
    
    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            
            # Check if too many recent failures
            ollama_perf = self.performance_history['ollama']
            if (ollama_perf.failure_count >= self.failure_threshold and 
                ollama_perf.last_failure and 
                datetime.now() - ollama_perf.last_failure < timedelta(seconds=self.recovery_timeout)):
                return False
            
            # Quick connectivity check
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
            
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {str(e)}")
            return False
    
    def _is_gemini_available(self) -> bool:
        """Check if Gemini is available (API key, rate limits, cost limits)"""
        try:
            # Check cost limits first
            if self.current_daily_cost >= self.daily_cost_limit:
                logger.info(f"Gemini unavailable: daily cost limit reached (${self.current_daily_cost:.2f})")
                return False
            
            from cloud_deployment.api.providers.gemini_provider import GeminiLLMProvider
            gemini = GeminiLLMProvider()
            
            # Check circuit breaker
            gemini_perf = self.performance_history['gemini']
            if (gemini_perf.failure_count >= self.failure_threshold and 
                gemini_perf.last_failure and 
                datetime.now() - gemini_perf.last_failure < timedelta(seconds=self.recovery_timeout)):
                return False
            
            return gemini.is_available()
            
        except Exception as e:
            logger.debug(f"Gemini availability check failed: {str(e)}")
            return False
    
    def _prefer_local(self, available_providers: List[str]) -> str:
        """Prefer local provider (privacy-first)"""
        if 'ollama' in available_providers:
            return 'ollama'
        elif 'gemini' in available_providers:
            logger.info("Local provider unavailable, falling back to Gemini")
            return 'gemini'
        else:
            raise Exception("No providers available")
    
    def _prefer_cloud(self, available_providers: List[str]) -> str:
        """Prefer cloud provider (performance-first)"""
        if 'gemini' in available_providers:
            return 'gemini'
        elif 'ollama' in available_providers:
            logger.info("Cloud provider unavailable, falling back to Ollama")
            return 'ollama'
        else:
            raise Exception("No providers available")
    
    def _prefer_cost_effective(self, available_providers: List[str]) -> str:
        """Prefer cost-effective provider"""
        # Always prefer free local provider
        if 'ollama' in available_providers:
            return 'ollama'
        elif 'gemini' in available_providers and self.current_daily_cost < self.daily_cost_limit:
            return 'gemini'
        else:
            raise Exception("No cost-effective providers available")
    
    def _smart_selection(self, available_providers: List[str], query: str, context: str) -> str:
        """Intelligent provider selection based on multiple factors"""
        
        # If only one provider available, use it
        if len(available_providers) == 1:
            return available_providers[0]
        
        # Calculate query complexity
        query_complexity = len(query.split()) + len(context.split()) / 4
        
        # Get performance metrics
        ollama_perf = self.performance_history['ollama']
        gemini_perf = self.performance_history['gemini']
        
        # Scoring system (higher is better)
        scores = {}
        
        for provider in available_providers:
            score = 0
            perf = self.performance_history[provider]
            
            # Success rate factor (0-40 points)
            score += perf.success_rate * 40
            
            # Response time factor (0-30 points, inverse relationship)
            if perf.avg_response_time > 0:
                # Faster response = higher score
                time_score = max(0, 30 - (perf.avg_response_time / 10))
                score += time_score
            else:
                score += 25  # Default for no history
            
            # Cost factor (0-20 points)
            if provider == 'ollama':
                score += 20  # Free
            elif provider == 'gemini':
                remaining_budget = self.daily_cost_limit - self.current_daily_cost
                cost_score = min(20, (remaining_budget / self.daily_cost_limit) * 20)
                score += cost_score
            
            # Complexity handling (0-10 points)
            if provider == 'gemini' and query_complexity > 50:
                score += 10  # Gemini better for complex queries
            elif provider == 'ollama' and query_complexity <= 20:
                score += 8   # Ollama good for simple queries
            
            scores[provider] = score
            logger.debug(f"{provider} score: {score:.1f}")
        
        # Select highest scoring provider
        best_provider = max(scores.keys(), key=lambda k: scores[k])
        
        logger.info(f"Smart selection: {best_provider} (score: {scores[best_provider]:.1f})")
        return best_provider
    
    def _execute_ollama_request(self, query: str, context: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Execute request using Ollama"""
        import requests
        
        # Build prompt similar to Gemini format
        prompt = self._build_rag_prompt(query, context, conversation_history)
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        result = response.json()
        
        return {
            'success': True,
            'response': result.get('response', ''),
            'token_count': len(result.get('response', '').split()) * 1.3,  # Approximate token count
            'provider': 'ollama'
        }
    
    def _execute_gemini_request(self, query: str, context: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Execute request using Gemini"""
        from cloud_deployment.api.providers.gemini_provider import GeminiLLMProvider
        
        gemini = GeminiLLMProvider()
        result = gemini.generate_rag_response(query, context, conversation_history)
        
        # Track cost (approximate)
        if result.get('success') and result.get('token_count'):
            cost = result['token_count'] * 0.000001  # Rough estimate
            self.current_daily_cost += cost
        
        return result
    
    def _build_rag_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Build RAG prompt for Ollama (similar to Gemini format)"""
        prompt_parts = [
            "You are a helpful AI assistant with access to a knowledge base.",
            "Provide educational, accurate responses based on the context provided.",
            ""
        ]
        
        if conversation_history:
            prompt_parts.append("Conversation History:")
            for entry in conversation_history[-3:]:  # Keep last 3 exchanges for Ollama
                prompt_parts.append(f"User: {entry.get('user', '')}")
                prompt_parts.append(f"Assistant: {entry.get('assistant', '')}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Relevant Context from Knowledge Base:",
            context,
            "",
            f"User Question: {query}",
            "",
            "Please provide a helpful response based on the context provided."
        ])
        
        return "\n".join(prompt_parts)
    
    def _try_fallback(self, failed_provider: str, query: str, context: str, conversation_history: List[Dict], error: str) -> Dict[str, Any]:
        """Try fallback provider when primary fails"""
        
        fallback_provider = 'gemini' if failed_provider == 'ollama' else 'ollama'
        
        available_providers = self._get_available_providers()
        if fallback_provider not in available_providers:
            return {
                'success': False,
                'error': f"Primary provider ({failed_provider}) failed: {error}. No fallback available.",
                'provider_used': failed_provider
            }
        
        logger.info(f"🔄 Trying fallback: {fallback_provider}")
        
        try:
            # Recursive call but with explicit provider
            start_time = time.time()
            
            if fallback_provider == 'ollama':
                result = self._execute_ollama_request(query, context, conversation_history)
            else:
                result = self._execute_gemini_request(query, context, conversation_history)
            
            response_time = time.time() - start_time
            self._track_success(fallback_provider, response_time, result.get('token_count', 0))
            
            result['provider_used'] = fallback_provider
            result['response_time'] = response_time
            result['fallback_used'] = True
            result['primary_failure'] = error
            
            return result
            
        except Exception as e:
            self._track_failure(fallback_provider, str(e))
            return {
                'success': False,
                'error': f"Both providers failed. Primary ({failed_provider}): {error}. Fallback ({fallback_provider}): {str(e)}",
                'provider_used': failed_provider
            }
    
    def _track_success(self, provider: str, response_time: float, token_count: int):
        """Track successful request"""
        perf = self.performance_history[provider]
        perf.total_requests += 1
        perf.total_tokens += token_count
        
        # Update success rate
        successful_requests = perf.total_requests - perf.failure_count
        perf.success_rate = successful_requests / perf.total_requests
        
        # Update average response time
        if perf.avg_response_time == 0:
            perf.avg_response_time = response_time
        else:
            perf.avg_response_time = (perf.avg_response_time * 0.8) + (response_time * 0.2)
    
    def _track_failure(self, provider: str, error: str):
        """Track failed request"""
        perf = self.performance_history[provider]
        perf.failure_count += 1
        perf.total_requests += 1
        perf.last_failure = datetime.now()
        
        # Update success rate
        successful_requests = perf.total_requests - perf.failure_count
        perf.success_rate = successful_requests / perf.total_requests if perf.total_requests > 0 else 0
    
    def _reset_daily_cost_if_needed(self):
        """Reset daily cost if it's a new day"""
        today = datetime.now().date()
        if today != self.cost_reset_date:
            self.current_daily_cost = 0.0
            self.cost_reset_date = today
            logger.info("Daily cost counter reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        return {
            'performance_history': {
                provider: {
                    'success_rate': perf.success_rate,
                    'avg_response_time': perf.avg_response_time,
                    'total_requests': perf.total_requests,
                    'total_tokens': perf.total_tokens,
                    'failure_count': perf.failure_count
                }
                for provider, perf in self.performance_history.items()
            },
            'current_daily_cost': self.current_daily_cost,
            'daily_cost_limit': self.daily_cost_limit,
            'user_preference': self.user_preference.value,
            'available_providers': self._get_available_providers()
        } 