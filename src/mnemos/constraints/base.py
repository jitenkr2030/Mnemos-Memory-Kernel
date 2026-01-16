"""
Mnemos Constraints Base Module - Layer 4

This module defines the base classes and interfaces for the domain
constraint system. All domain-specific validators must implement
these interfaces to integrate with the Mnemos kernel.

The constraint system provides:
- Abstract base class for all constraints
- Result types for validation outcomes
- Registry for managing constraint plugins
- Support for different validation severities
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Type
from abc import ABC, abstractmethod


class ConstraintType(str, Enum):
    """
    Types of constraints that can be applied.
    
    Each constraint type represents a different category of validation:
    - DATA_INTEGRITY: Ensures data consistency and correctness
    - BUSINESS_RULE: Enforces domain-specific business logic
    - TEMPORAL: Validates time-related constraints
    - ENTITY_VALIDATION: Specialized entity extraction and validation
    - FORMAT: Ensures proper formatting of data
    - SEMANTIC: Validates semantic correctness of memories
    """
    DATA_INTEGRITY = "data_integrity"
    BUSINESS_RULE = "business_rule"
    TEMPORAL = "temporal"
    ENTITY_VALIDATION = "entity_validation"
    FORMAT = "format"
    SEMANTIC = "semantic"


class ValidationSeverity(str, Enum):
    """
    Severity levels for constraint violations.
    
    Different severities determine how violations are handled:
    - ERROR: Prevents memory from being stored
    - WARNING: Stores memory but logs a warning
    - INFO: Informational feedback only
    - SUGGESTION: Provides improvement suggestions
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUGGESTION = "suggestion"


@dataclass
class ConstraintResult:
    """
    Result of a constraint validation check.
    
    This class represents the outcome of validating a constraint
    against a memory node. It includes the validation status,
    any violations found, and suggestions for correction.
    
    Attributes:
        constraint_name: Name of the constraint that was checked
        passed: Whether the validation passed
        severity: Severity level of any violations
        message: Human-readable validation message
        violations: List of specific violations found
        suggestions: Suggestions for fixing violations
        metadata: Additional context about the validation
        timestamp: When the validation was performed
    """
    constraint_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "constraint_name": self.constraint_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "violations": self.violations,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def passed_result(
        cls,
        constraint_name: str,
        message: str = "Constraint validation passed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConstraintResult":
        """Create a passed validation result."""
        return cls(
            constraint_name=constraint_name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message=message,
            metadata=metadata or {}
        )
    
    @classmethod
    def failed_result(
        cls,
        constraint_name: str,
        message: str,
        severity: ValidationSeverity,
        violations: List[str] = None,
        suggestions: List[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ConstraintResult":
        """Create a failed validation result."""
        return cls(
            constraint_name=constraint_name,
            passed=False,
            severity=severity,
            message=message,
            violations=violations or [],
            suggestions=suggestions or [],
            metadata=metadata or {}
        )


class BaseConstraint(ABC):
    """
    Abstract base class for all domain constraints.
    
    All domain-specific constraints must inherit from this class
    and implement the required methods. The constraint system uses
    polymorphism to apply different validation rules based on the
    domain of the memory being processed.
    
    Subclasses should implement:
    - validate(): Perform the actual validation
    - name: Unique identifier for the constraint
    - constraint_type: Category of the constraint
    - description: Human-readable description
    
    Example implementation:
    
    ```python
    class GSTValidator(BaseConstraint):
        \"\"\"Validates GST-related information in memories.\"\"\"
        
        @property
        def name(self) -> str:
            return "gst_validator"
        
        @property
        def constraint_type(self) -> ConstraintType:
            return ConstraintType.BUSINESS_RULE
        
        @property
        def description(self) -> str:
            return "Validates GST numbers and calculations in accounting contexts"
        
        def validate(self, memory: MemoryNode) -> ConstraintResult:
            # Implementation here
            pass
    ```
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this constraint.
        
        Returns:
            A lowercase, snake_case string uniquely identifying the constraint
        """
        pass
    
    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType:
        """
        Category of this constraint.
        
        Returns:
            The ConstraintType enum value representing the category
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of this constraint.
        
        Returns:
            A clear description of what the constraint validates
        """
        pass
    
    @property
    def version(self) -> str:
        """
        Version of this constraint.
        
        Override this property to track constraint versions.
        """
        return "1.0.0"
    
    @property
    def enabled(self) -> bool:
        """
        Whether this constraint is enabled by default.
        
        Override to change the default enabled state.
        """
        return True
    
    @abstractmethod
    def validate(self, memory) -> ConstraintResult:
        """
        Validate a memory against this constraint.
        
        This method is called for each memory that should be
        validated against this constraint. The implementation
        should examine the memory and return a ConstraintResult
        indicating whether validation passed or failed.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        pass
    
    def validate_batch(self, memories: list) -> List[ConstraintResult]:
        """
        Validate multiple memories at once.
        
        Override this method for efficiency when validating
        multiple memories, as some constraints can batch
        their validation for better performance.
        
        Args:
            memories: List of MemoryNodes to validate
            
        Returns:
            List of ConstraintResults, one per memory
        """
        return [self.validate(memory) for memory in memories]
    
    def before_ingest(self, memory) -> Optional[ConstraintResult]:
        """
        Hook called before a memory is ingested.
        
        This method is called during the ingestion pipeline,
        before the memory is stored. It can be used to:
        - Modify the memory before storage
        - Reject the memory by returning a failed result
        - Add annotations to the memory
        
        Args:
            memory: The MemoryNode being ingested
            
        Returns:
            None to continue processing, or a ConstraintResult to halt/modify
        """
        return None
    
    def after_ingest(self, memory) -> Optional[ConstraintResult]:
        """
        Hook called after a memory is ingested.
        
        This method is called after the memory has been stored.
        It can be used to:
        - Perform post-validation
        - Trigger side effects
        - Update related data structures
        
        Args:
            memory: The MemoryNode that was ingested
            
        Returns:
            None to continue, or a ConstraintResult for additional info
        """
        return None
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this constraint.
        
        Returns:
            Dictionary describing the configuration options
        """
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether this constraint is enabled",
                    "default": self.enabled
                }
            }
        }
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure this constraint with provided options.
        
        Args:
            config: Configuration dictionary
        """
        if "enabled" in config:
            # This would typically modify the enabled state
            pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.constraint_type.value})"


class ConstraintRegistry:
    """
    Registry for managing constraint plugins.
    
    The registry provides a central location for registering,
    retrieving, and managing domain constraints. It supports:
    - Registering constraints by name and type
    - Looking up constraints
    - Filtering constraints by type
    - Managing constraint lifecycle
    
    This is a singleton-style registry that can be used to
    manage constraints across the application.
    """
    
    _instance: Optional["ConstraintRegistry"] = None
    _constraints: Dict[str, BaseConstraint] = {}
    _constraints_by_type: Dict[ConstraintType, List[str]] = {}
    
    def __new__(cls) -> "ConstraintRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the registry with empty collections."""
        self._constraints = {}
        self._constraints_by_type = {}
        for constraint_type in ConstraintType:
            self._constraints_by_type[constraint_type] = []
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._instance = None
    
    def register(self, constraint: BaseConstraint) -> bool:
        """
        Register a constraint with the registry.
        
        Args:
            constraint: The constraint to register
            
        Returns:
            True if registration was successful
        """
        name = constraint.name
        
        if name in self._constraints:
            return False
        
        self._constraints[name] = constraint
        
        # Also register by type
        if constraint.constraint_type not in self._constraints_by_type:
            self._constraints_by_type[constraint.constraint_type] = []
        self._constraints_by_type[constraint.constraint_type].append(name)
        
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a constraint by name.
        
        Args:
            name: Name of the constraint to unregister
            
        Returns:
            True if unregistration was successful
        """
        if name not in self._constraints:
            return False
        
        constraint = self._constraints[name]
        constraint_type = constraint.constraint_type
        
        # Remove from type index
        if name in self._constraints_by_type[constraint_type]:
            self._constraints_by_type[constraint_type].remove(name)
        
        # Remove from main registry
        del self._constraints[name]
        
        return True
    
    def get(self, name: str) -> Optional[BaseConstraint]:
        """
        Retrieve a constraint by name.
        
        Args:
            name: Name of the constraint
            
        Returns:
            The constraint if found, None otherwise
        """
        return self._constraints.get(name)
    
    def get_by_type(self, constraint_type: ConstraintType) -> List[BaseConstraint]:
        """
        Retrieve all constraints of a given type.
        
        Args:
            constraint_type: The type of constraints to retrieve
            
        Returns:
            List of constraints of the specified type
        """
        names = self._constraints_by_type.get(constraint_type, [])
        return [self._constraints[name] for name in names if name in self._constraints]
    
    def get_all(self) -> List[BaseConstraint]:
        """
        Retrieve all registered constraints.
        
        Returns:
            List of all registered constraints
        """
        return list(self._constraints.values())
    
    def get_enabled(self) -> List[BaseConstraint]:
        """
        Retrieve all enabled constraints.
        
        Returns:
            List of enabled constraints
        """
        return [c for c in self._constraints.values() if c.enabled]
    
    def count(self) -> int:
        """
        Get the number of registered constraints.
        
        Returns:
            Total number of constraints
        """
        return len(self._constraints)
    
    def clear(self) -> None:
        """Remove all registered constraints."""
        self._constraints = {}
        self._initialize()
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about the registry state.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_constraints": self.count(),
            "by_type": {
                ct.value: len(names)
                for ct, names in self._constraints_by_type.items()
            },
            "enabled_count": len(self.get_enabled())
        }
