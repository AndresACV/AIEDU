"""
Configuration Manager for Dual Architecture AIEDU System

Handles configuration loading and management for both local and cloud deployments
with automatic environment detection and validation.
"""

import os
import yaml
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path


class DeploymentMode(Enum):
    """Deployment modes for the AIEDU system."""
    LOCAL = "local"
    CLOUD = "cloud"


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigManager:
    """
    Manages configuration for dual architecture deployment.
    
    Automatically detects deployment mode and environment,
    loads appropriate configuration files, and provides
    unified access to settings.
    """
    
    def __init__(self, 
                 deployment_mode: Optional[DeploymentMode] = None,
                 environment: Optional[Environment] = None,
                 config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            deployment_mode: Force specific deployment mode (auto-detect if None)
            environment: Force specific environment (auto-detect if None)
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.deployment_mode = deployment_mode or self._detect_deployment_mode()
        self.environment = environment or self._detect_environment()
        self.config = self._load_config()
        
    def _detect_deployment_mode(self) -> DeploymentMode:
        """Auto-detect deployment mode based on environment variables."""
        # Check for explicit deployment mode setting
        mode = os.getenv('DEPLOYMENT_MODE', '').lower()
        if mode == 'cloud':
            return DeploymentMode.CLOUD
        elif mode == 'local':
            return DeploymentMode.LOCAL
            
        # Check for cloud indicators
        if any([
            os.getenv('VERCEL'),                    # Vercel deployment
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS'), # Google Cloud
            os.getenv('GEMINI_API_KEY'),           # Gemini API
            os.getenv('PINECONE_API_KEY'),         # Pinecone
        ]):
            return DeploymentMode.CLOUD
            
        # Check for local indicators
        if any([
            os.path.exists('local_deployment/web_app'),  # Local structure
            os.getenv('OLLAMA_HOST'),              # Ollama
            os.path.exists('vector_db'),           # Local ChromaDB
        ]):
            return DeploymentMode.LOCAL
            
        # Default to local for development
        return DeploymentMode.LOCAL
    
    def _detect_environment(self) -> Environment:
        """Auto-detect environment based on various indicators."""
        # Check explicit environment setting
        env = os.getenv('ENVIRONMENT', '').lower()
        if env in ['production', 'prod']:
            return Environment.PRODUCTION
        elif env in ['staging', 'stage']:
            return Environment.STAGING
        elif env in ['development', 'dev']:
            return Environment.DEVELOPMENT
            
        # Check for production indicators
        if any([
            os.getenv('VERCEL_ENV') == 'production',
            os.getenv('NODE_ENV') == 'production',
            not os.getenv('DEBUG', '').lower() in ['true', '1'],
        ]):
            return Environment.PRODUCTION
            
        # Default to development
        return Environment.DEVELOPMENT
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        # Load base configuration for deployment mode
        base_config_file = self.config_dir / f"{self.deployment_mode.value}.yaml"
        
        if not base_config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {base_config_file}"
            )
            
        with open(base_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment-specific settings if available
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f)
                config = self._merge_configs(config, env_config)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Apply API keys and credentials from environment
        providers = config.get('providers', {})
        
        # LLM provider overrides
        if 'llm' in providers:
            llm_config = providers['llm']
            if llm_config.get('type') == 'gemini':
                api_key_env = llm_config.get('api_key_env')
                if api_key_env and os.getenv(api_key_env):
                    llm_config['api_key'] = os.getenv(api_key_env)
        
        # STT provider overrides
        if 'stt' in providers:
            stt_config = providers['stt']
            if stt_config.get('type') == 'google_cloud_stt':
                creds_env = stt_config.get('credentials_env')
                if creds_env and os.getenv(creds_env):
                    stt_config['credentials_path'] = os.getenv(creds_env)
        
        # TTS provider overrides
        if 'tts' in providers:
            tts_config = providers['tts']
            if tts_config.get('type') == 'google_cloud_tts':
                creds_env = tts_config.get('credentials_env')
                if creds_env and os.getenv(creds_env):
                    tts_config['credentials_path'] = os.getenv(creds_env)
        
        # Vector store overrides
        if 'vector_store' in providers:
            vs_config = providers['vector_store']
            if vs_config.get('type') == 'pinecone':
                api_key_env = vs_config.get('api_key_env')
                env_env = vs_config.get('environment_env')
                if api_key_env and os.getenv(api_key_env):
                    vs_config['api_key'] = os.getenv(api_key_env)
                if env_env and os.getenv(env_env):
                    vs_config['environment'] = os.getenv(env_env)
        
        return config
    
    def get_provider_config(self, service_type: str) -> Dict[str, Any]:
        """Get configuration for a specific provider service."""
        providers = self.config.get('providers', {})
        if service_type not in providers:
            raise KeyError(f"Provider configuration not found for: {service_type}")
        return providers[service_type]
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.config.get('app', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get('performance', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.config.get('security', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def is_local_deployment(self) -> bool:
        """Check if this is a local deployment."""
        return self.deployment_mode == DeploymentMode.LOCAL
    
    def is_cloud_deployment(self) -> bool:
        """Check if this is a cloud deployment."""
        return self.deployment_mode == DeploymentMode.CLOUD
    
    def is_production(self) -> bool:
        """Check if this is a production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if this is a development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def validate_config(self) -> bool:
        """Validate configuration completeness and correctness."""
        errors = []
        
        # Check required provider configurations
        providers = self.config.get('providers', {})
        required_providers = ['llm', 'stt', 'tts', 'vector_store', 'embeddings']
        
        for provider in required_providers:
            if provider not in providers:
                errors.append(f"Missing provider configuration: {provider}")
            else:
                provider_config = providers[provider]
                if 'type' not in provider_config:
                    errors.append(f"Missing 'type' in {provider} configuration")
        
        # Check cloud-specific requirements
        if self.is_cloud_deployment():
            cloud_requirements = [
                ('GEMINI_API_KEY', 'Gemini API key'),
                ('GOOGLE_APPLICATION_CREDENTIALS', 'Google Cloud credentials'),
                ('PINECONE_API_KEY', 'Pinecone API key'),
                ('PINECONE_ENVIRONMENT', 'Pinecone environment'),
            ]
            
            for env_var, description in cloud_requirements:
                if not os.getenv(env_var):
                    errors.append(f"Missing required environment variable: {env_var} ({description})")
        
        # Check local-specific requirements
        if self.is_local_deployment():
            local_requirements = [
                ('local_deployment/web_app', 'Local web application'),
                ('vector_db', 'Local vector database'),
            ]
            
            for path, description in local_requirements:
                if not os.path.exists(path):
                    errors.append(f"Missing local requirement: {path} ({description})")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'deployment_mode': self.deployment_mode.value,
            'environment': self.environment.value,
            'providers': {
                name: config.get('type', 'unknown') 
                for name, config in self.config.get('providers', {}).items()
            },
            'app_config': {
                'host': self.config.get('app', {}).get('host'),
                'port': self.config.get('app', {}).get('port'),
                'debug': self.config.get('app', {}).get('debug'),
            },
            'config_file': f"{self.deployment_mode.value}.yaml",
            'validated': self.validate_config()
        }


# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config():
    """Reload configuration (useful for testing)."""
    global _config_manager
    _config_manager = None
    return get_config_manager()


if __name__ == "__main__":
    # Example usage and testing
    config = ConfigManager()
    
    print("Configuration Summary:")
    print(yaml.dump(config.get_config_summary(), default_flow_style=False))
    
    print("\nValidation Result:")
    is_valid = config.validate_config()
    print(f"Configuration is {'valid' if is_valid else 'invalid'}") 