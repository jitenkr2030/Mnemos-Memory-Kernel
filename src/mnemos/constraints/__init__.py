"""
Mnemos Constraints Module - Layer 4

This module provides domain-specific constraint validation and rules
for the Mnemos memory system. It enables specialized validation for
different domains such as accounting, legal, healthcare, and more.

Key components:
- Base constraint classes and interfaces
- Plugin system for domain-specific validators
- Truth rules enforcement
- Specialized entity extraction and validation
"""

from .base import (
    BaseConstraint,
    ConstraintResult,
    ConstraintType,
    ValidationSeverity,
    ConstraintRegistry,
)
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
from .constraint_engine import ConstraintEngine, ConstraintEngineResult

__all__ = [
    # Base classes
    "BaseConstraint",
    "ConstraintResult",
    "ConstraintType",
    "ValidationSeverity",
    "ConstraintRegistry",
    # Validators
    "GSTValidator",
    "InvoiceValidator",
    "DateConsistencyValidator",
    "EmailValidator",
    "URLValidator",
    "CurrencyValidator",
    "PhoneNumberValidator",
    "BusinessRuleValidator",
    # Engine
    "ConstraintEngine",
    "ConstraintEngineResult",
]
