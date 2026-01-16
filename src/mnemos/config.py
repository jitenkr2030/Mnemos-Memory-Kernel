"""
Mnemos Configuration Module

Provides configuration management for the Mnemos memory kernel.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MnemosConfig:
    """
    Configuration for the Mnemos memory kernel.
    
    Attributes:
        storage_dir: Directory for memory storage
        enable_llm_classification: Whether to use LLM for intent classification
        llm_provider: LLM provider to use
        enable_evolution: Whether to enable evolution intelligence
        enable_recall: Whether to enable recall engine
        recall_insights: Whether to generate insights during recall
        recall_limit: Default limit for recall results
        enable_constraints: Whether to enable domain constraints
        constraints_fail_on_error: Whether to fail on constraint errors
    """
    storage_dir: str = "./data"
    enable_llm_classification: bool = False
    llm_provider: Optional[str] = None
    enable_evolution: bool = True
    enable_recall: bool = True
    recall_insights: bool = True
    recall_limit: int = 20
    enable_constraints: bool = True
    constraints_fail_on_error: bool = False
    
    @classmethod
    def from_env(cls) -> "MnemosConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
        - MNEMOS_STORAGE_DIR: Storage directory
        - MNEMOS_LLM_ENABLED: Enable LLM classification
        - MNEMOS_LLM_PROVIDER: LLM provider name
        - MNEMOS_EVOLUTION_ENABLED: Enable evolution intelligence
        - MNEMOS_RECALL_ENABLED: Enable recall engine
        - MNEMOS_RECALL_INSIGHTS: Generate recall insights
        - MNEMOS_RECALL_LIMIT: Default recall limit
        - MNEMOS_CONSTRAINTS_ENABLED: Enable constraints
        - MNEMOS_CONSTRAINTS_FAIL_ON_ERROR: Fail on constraint errors
        """
        import os
        
        return cls(
            storage_dir=os.getenv("MNEMOS_STORAGE_DIR", "./data"),
            enable_llm_classification=os.getenv("MNEMOS_LLM_ENABLED", "").lower() == "true",
            llm_provider=os.getenv("MNEMOS_LLM_PROVIDER"),
            enable_evolution=os.getenv("MNEMOS_EVOLUTION_ENABLED", "true").lower() == "true",
            enable_recall=os.getenv("MNEMOS_RECALL_ENABLED", "true").lower() == "true",
            recall_insights=os.getenv("MNEMOS_RECALL_INSIGHTS", "true").lower() == "true",
            recall_limit=int(os.getenv("MNEMOS_RECALL_LIMIT", "20")),
            enable_constraints=os.getenv("MNEMOS_CONSTRAINTS_ENABLED", "true").lower() == "true",
            constraints_fail_on_error=os.getenv("MNEMOS_CONSTRAINTS_FAIL_ON_ERROR", "false").lower() == "true"
        )
    
    def to_kernel_kwargs(self) -> dict:
        """
        Convert configuration to keyword arguments for kernel initialization.
        
        Returns:
            Dictionary of kernel constructor arguments
        """
        return {
            "storage_dir": self.storage_dir,
            "enable_llm_classification": self.enable_llm_classification,
            "llm_provider": self.llm_provider,
            "enable_evolution": self.enable_evolution,
            "enable_recall": self.enable_recall,
            "recall_insights": self.recall_insights,
            "recall_limit": self.recall_limit,
            "enable_constraints": self.enable_constraints,
            "constraints_fail_on_error": self.constraints_fail_on_error
        }


# Default configuration instance
default_config = MnemosConfig()
