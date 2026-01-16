"""
Configuration Management for Mnemos

This module provides centralized configuration management for the Mnemos
memory kernel. Configuration can be loaded from environment variables,
configuration files, or passed directly during initialization.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MnemosConfig:
    """
    Central configuration for the Mnemos kernel.
    
    This configuration class holds all configurable parameters for the
    memory kernel system. Values can be set programmatically, loaded from
    environment variables, or read from configuration files.
    
    Attributes:
        storage_dir: Directory for storing memory data
        llm_enabled: Whether to enable LLM-based classification
        llm_provider: LLM provider name (openai, anthropic, local)
        embedding_model: Model name for generating embeddings
        log_level: Logging level (debug, info, warning, error)
        api_host: Host for the API server
        api_port: Port for the API server
    """
    storage_dir: str = "./data"
    llm_enabled: bool = False
    llm_provider: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: str = "info"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Additional configuration storage
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, prefix: str = "MNEMOS_") -> "MnemosConfig":
        """
        Load configuration from environment variables.
        
        Environment variables are converted from uppercase with underscores
        to the corresponding configuration attribute.
        
        Example: MNEMOS_STORAGE_DIR -> storage_dir
        """
        config = cls()
        
        env_mappings = {
            f"{prefix}STORAGE_DIR": "storage_dir",
            f"{prefix}LLM_ENABLED": "llm_enabled",
            f"{prefix}LLM_PROVIDER": "llm_provider",
            f"{prefix}EMBEDDING_MODEL": "embedding_model",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}API_HOST": "api_host",
            f"{prefix}API_PORT": "api_port",
        }
        
        for env_var, attr_name in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if attr_name in ("llm_enabled",):
                    config.set_attr(attr_name, value.lower() in ("true", "1", "yes"))
                elif attr_name in ("api_port",):
                    config.set_attr(attr_name, int(value))
                else:
                    config.set_attr(attr_name, value)
        
        return config
    
    def set_attr(self, name: str, value: Any) -> None:
        """Set a configuration attribute dynamically."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.extra[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "storage_dir": self.storage_dir,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "embedding_model": self.embedding_model,
            "log_level": self.log_level,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "extra": self.extra
        }


# Default configuration instance
default_config = MnemosConfig()
