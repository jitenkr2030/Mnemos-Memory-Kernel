"""
Mnemos Constraint Engine - Layer 4

This module implements the Constraint Engine that orchestrates
all domain constraints during the memory lifecycle. The engine
manages constraint registration, validation execution, and result
aggregation.

The constraint engine provides:
- Centralized constraint management
- Batch validation support
- Integration with the memory ingestion pipeline
- Detailed validation reporting
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from .base import (
    BaseConstraint,
    ConstraintResult,
    ConstraintType,
    ValidationSeverity,
    ConstraintRegistry,
)
from ..kernel.memory_node import MemoryNode


@dataclass
class ConstraintEngineResult:
    """
    Result of a constraint validation operation.
    
    This class represents the complete outcome of validating
    memories against all registered constraints. It includes
    aggregated results, statistics, and recommendations.
    
    Attributes:
        overall_passed: Whether all critical validations passed
        memory_id: ID of the validated memory
        results: List of individual constraint results
        violation_count: Total number of violations found
        warning_count: Number of warnings
        error_count: Number of errors
        execution_time_ms: Time taken for validation
        recommendations: List of recommendations based on results
    """
    overall_passed: bool
    memory_id: str
    results: List[ConstraintResult] = field(default_factory=list)
    violation_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    execution_time_ms: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_passed": self.overall_passed,
            "memory_id": self.memory_id,
            "total_results": len(self.results),
            "violation_count": self.violation_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "execution_time_ms": self.execution_time_ms,
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata
        }
    
    def get_results_by_severity(self, severity: ValidationSeverity) -> List[ConstraintResult]:
        """Get all results of a specific severity."""
        return [r for r in self.results if r.severity == severity]
    
    def get_failed_results(self) -> List[ConstraintResult]:
        """Get all failed validation results."""
        return [r for r in self.results if not r.passed]
    
    def get_result_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation results."""
        return {
            "passed": self.overall_passed,
            "violations": self.violation_count,
            "warnings": self.warning_count,
            "errors": self.error_count,
            "constraints_checked": len(self.results)
        }


class ConstraintEngine:
    """
    Main engine for managing and executing domain constraints.
    
    The constraint engine is responsible for:
    - Managing registered constraints
    - Executing validation during memory lifecycle
    - Aggregating and reporting validation results
    - Integrating with the Mnemos kernel
    
    The engine can be configured with:
    - Enabled/disabled state
    - Error handling behavior (fail fast, collect all)
    - Custom constraint sets
    
    Attributes:
        registry: Constraint registry for managing constraints
        enabled: Whether the engine is active
        fail_on_error: Whether to stop on first error
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        enabled: bool = True,
        fail_on_error: bool = False,
        auto_register_defaults: bool = True
    ):
        """
        Initialize the constraint engine.
        
        Args:
            storage_dir: Directory for constraint engine data
            enabled: Whether the engine is active
            fail_on_error: Whether to stop validation on first error
            auto_register_defaults: Whether to register default validators
        """
        self.registry = ConstraintRegistry()
        self.enabled = enabled
        self.fail_on_error = fail_on_error
        self.storage_dir = Path(storage_dir)
        
        # Statistics tracking
        self._validation_count = 0
        self._total_execution_time = 0.0
        
        # Register default constraints
        if auto_register_defaults:
            self._register_default_constraints()
    
    def _register_default_constraints(self) -> None:
        """Register the default set of constraint validators."""
        from .validators import (
            GSTValidator,
            InvoiceValidator,
            DateConsistencyValidator,
            EmailValidator,
            URLValidator,
            CurrencyValidator,
            PhoneNumberValidator,
            BusinessRuleValidator,
        )
        
        default_constraints = [
            DateConsistencyValidator(),
            EmailValidator(),
            URLValidator(),
            CurrencyValidator(),
            PhoneNumberValidator(),
            BusinessRuleValidator(),
        ]
        
        for constraint in default_constraints:
            self.register_constraint(constraint)
    
    def register_constraint(self, constraint: BaseConstraint) -> bool:
        """
        Register a constraint with the engine.
        
        Args:
            constraint: The constraint to register
            
        Returns:
            True if registration was successful
        """
        return self.registry.register(constraint)
    
    def unregister_constraint(self, name: str) -> bool:
        """
        Unregister a constraint by name.
        
        Args:
            name: Name of the constraint to remove
            
        Returns:
            True if unregistration was successful
        """
        return self.registry.unregister(name)
    
    def get_constraint(self, name: str) -> Optional[BaseConstraint]:
        """
        Retrieve a constraint by name.
        
        Args:
            name: Name of the constraint
            
        Returns:
            The constraint if found
        """
        return self.registry.get(name)
    
    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[BaseConstraint]:
        """
        Get all constraints of a specific type.
        
        Args:
            constraint_type: Type of constraints to retrieve
            
        Returns:
            List of constraints of the specified type
        """
        return self.registry.get_by_type(constraint_type)
    
    def validate_memory(self, memory: MemoryNode) -> ConstraintEngineResult:
        """
        Validate a memory against all registered constraints.
        
        This is the primary method for constraint validation. It
        executes all registered constraints against the memory and
        returns an aggregated result.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintEngineResult with aggregated validation outcome
        """
        import time
        start_time = time.time()
        
        if not self.enabled:
            return ConstraintEngineResult(
                overall_passed=True,
                memory_id=memory.id,
                metadata={"reason": "Constraint engine disabled"}
            )
        
        results: List[ConstraintResult] = []
        violation_count = 0
        warning_count = 0
        error_count = 0
        
        # Get enabled constraints
        constraints = self.registry.get_enabled()
        
        for constraint in constraints:
            try:
                result = constraint.validate(memory)
                results.append(result)
                
                # Count violations
                if not result.passed:
                    violation_count += 1
                    if result.severity == ValidationSeverity.ERROR:
                        error_count += 1
                    elif result.severity == ValidationSeverity.WARNING:
                        warning_count += 1
                    
                    # Stop on error if configured
                    if self.fail_on_error and result.severity == ValidationSeverity.ERROR:
                        break
                        
            except Exception as e:
                # Log error but continue with other constraints
                results.append(ConstraintResult(
                    constraint_name=constraint.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Constraint validation error: {str(e)}",
                    violations=[f"Exception during validation: {str(e)}"]
                ))
        
        execution_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self._validation_count += 1
        self._total_execution_time += execution_time
        
        # Determine overall pass/fail
        overall_passed = error_count == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return ConstraintEngineResult(
            overall_passed=overall_passed,
            memory_id=memory.id,
            results=results,
            violation_count=violation_count,
            warning_count=warning_count,
            error_count=error_count,
            execution_time_ms=execution_time,
            recommendations=recommendations
        )
    
    def validate_batch(
        self,
        memories: List[MemoryNode],
        parallel: bool = False
    ) -> List[ConstraintEngineResult]:
        """
        Validate multiple memories at once.
        
        Args:
            memories: List of MemoryNodes to validate
            parallel: Whether to validate in parallel (for large batches)
            
        Returns:
            List of ConstraintEngineResults
        """
        if parallel:
            # Use concurrent validation for large batches
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.validate_memory, memories))
        else:
            results = [self.validate_memory(memory) for memory in memories]
        
        return results
    
    def before_ingest(self, memory: MemoryNode) -> Tuple[bool, Optional[ConstraintEngineResult]]:
        """
        Hook called before memory ingestion.
        
        This method can be used to perform pre-ingestion validation
        and potentially modify or reject the memory.
        
        Args:
            memory: The MemoryNode being ingested
            
        Returns:
            Tuple of (should_continue, validation_result)
        """
        if not self.enabled:
            return True, None
        
        result = self.validate_memory(memory)
        
        # Check if there are critical errors
        should_continue = result.error_count == 0
        
        return should_continue, result
    
    def after_ingest(
        self,
        memory: MemoryNode
    ) -> Optional[ConstraintEngineResult]:
        """
        Hook called after memory ingestion.
        
        This method can be used for post-validation and
        side effects.
        
        Args:
            memory: The MemoryNode that was ingested
            
        Returns:
            Validation result if any issues were found
        """
        if not self.enabled:
            return None
        
        result = self.validate_memory(memory)
        
        # Return result if there are any issues
        if result.violation_count > 0:
            return result
        
        return None
    
    def _generate_recommendations(self, results: List[ConstraintResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze failed results
        failed_results = [r for r in results if not r.passed]
        
        if not failed_results:
            return recommendations
        
        # Group by constraint type
        by_type: Dict[ConstraintType, int] = {}
        for result in failed_results:
            constraint = self.registry.get(result.constraint_name)
            if constraint:
                if constraint.constraint_type not in by_type:
                    by_type[constraint.constraint_type] = 0
                by_type[constraint.constraint_type] += 1
        
        # Generate type-specific recommendations
        if ConstraintType.FORMAT in by_type:
            recommendations.append("Consider reviewing the format of your memories for consistency")
        
        if ConstraintType.TEMPORAL in by_type:
            recommendations.append("Check timestamps and date references in memories")
        
        if ConstraintType.BUSINESS_RULE in by_type:
            recommendations.append("Review business rules and ensure all required information is captured")
        
        if ConstraintType.DATA_INTEGRITY in by_type:
            recommendations.append("Verify data integrity issues before proceeding")
        
        # Count repeated violations
        violation_messages = {}
        for result in failed_results:
            for violation in result.violations:
                if violation not in violation_messages:
                    violation_messages[violation] = 0
                violation_messages[violation] += 1
        
        # Suggest fixing common violations
        for violation, count in violation_messages.items():
            if count > 1:
                recommendations.append(
                    f"The violation '{violation[:50]}...' occurred {count} times. "
                    "Consider addressing the root cause."
                )
        
        return recommendations
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about constraint validation.
        
        Returns:
            Dictionary with validation statistics
        """
        avg_time = (
            self._total_execution_time / self._validation_count
            if self._validation_count > 0 else 0
        )
        
        return {
            "total_validations": self._validation_count,
            "total_execution_time_ms": self._total_execution_time,
            "average_execution_time_ms": avg_time,
            "constraints_registered": self.registry.count(),
            "constraints_enabled": len(self.registry.get_enabled()),
            "registry_info": self.registry.get_registry_info()
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get the current status of the constraint engine.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            "enabled": self.enabled,
            "fail_on_error": self.fail_on_error,
            "constraints_count": self.registry.count(),
            "constraints_by_type": {
                ct.value: len(self.registry.get_by_type(ct))
                for ct in ConstraintType
            },
            "validation_stats": self.get_validation_stats()
        }
    
    def clear_constraints(self) -> None:
        """Remove all registered constraints."""
        self.registry.clear()
    
    def enable(self) -> None:
        """Enable the constraint engine."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the constraint engine."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if the engine is enabled."""
        return self.enabled
