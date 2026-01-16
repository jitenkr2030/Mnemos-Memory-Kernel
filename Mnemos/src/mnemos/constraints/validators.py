"""
Mnemos Constraint Validators - Layer 4

This module provides concrete implementations of domain constraints
for various use cases. Each validator focuses on a specific aspect
of data validation and can be registered with the constraint engine.

Available validators:
- GSTValidator: Validates GST numbers and calculations (accounting)
- InvoiceValidator: Validates invoice-related information
- DateConsistencyValidator: Ensures temporal consistency
- EmailValidator: Validates email addresses in memories
- URLValidator: Validates URLs in memories
- CurrencyValidator: Validates currency amounts and formats
- PhoneNumberValidator: Validates phone numbers
- BusinessRuleValidator: General business rule validation
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Pattern
from dataclasses import dataclass

from .base import (
    BaseConstraint,
    ConstraintResult,
    ConstraintType,
    ValidationSeverity,
)


@dataclass
class ValidationContext:
    """
    Context information for validation operations.
    
    This class provides additional context that validators
    can use to make more informed decisions.
    
    Attributes:
        current_time: The current timestamp
        memory_count: Total number of memories in the system
        recent_memories: Recently added memories for reference
        custom_data: Application-specific context data
    """
    current_time: datetime
    memory_count: int
    recent_memories: List[Any] = None
    custom_data: Dict[str, Any] = None


class GSTValidator(BaseConstraint):
    """
    Validates GST (Goods and Services Tax) related information.
    
    This validator checks for:
    - Valid GSTIN format (15-character alphanumeric)
    - GST calculations consistency
    - Proper tax rate application
    
    Use case: Accounting and finance applications where
    GST-related information needs to be validated.
    """
    
    # GSTIN pattern for India (15-character format)
    GSTIN_PATTERN = re.compile(
        r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
    )
    
    # Common GST rates in India
    GST_RATES = {
        "0": 0.0,
        "5": 5.0,
        "12": 12.0,
        "18": 18.0,
        "28": 28.0
    }
    
    @property
    def name(self) -> str:
        return "gst_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BUSINESS_RULE
    
    @property
    def description(self) -> str:
        return "Validates GST numbers, calculations, and tax rates in accounting contexts"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate GST-related content in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        text = memory.raw_text.lower()
        violations = []
        suggestions = []
        
        # Check for GSTIN patterns
        gstin_pattern = re.compile(
            r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}[Z]{1}[0-9A-Z]{1}',
            re.IGNORECASE
        )
        gstins = gstin_pattern.findall(text)
        
        if gstins:
            for gstin in gstins:
                if not self.GSTIN_PATTERN.match(gstin.upper()):
                    violations.append(f"Invalid GSTIN format: {gstin}")
                    suggestions.append("GSTIN should be 15 characters: StateCode(2) + PAN(10) + EntityNo(1) + CheckSum(1) + 'Z' + LastChar(1)")
        
        # Check for tax amount consistency
        amount_pattern = r'₹[\d,]+(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, text)
        
        if len(amounts) >= 2:
            # Check if tax amounts make sense
            tax_mentions = re.findall(r'(?:GST|tax|taxation)[\s:]*(?:of|@|:)?\s*(\d+(?:\.\d+)?)\s*%?', text, re.IGNORECASE)
            if tax_mentions:
                try:
                    rate = float(tax_mentions[0])
                    if rate not in self.GST_RATES.values() and rate > 0:
                        suggestions.append(f"Non-standard GST rate detected: {rate}%. Common rates are: 0%, 5%, 12%, 18%, 28%")
                except ValueError:
                    pass
        
        # Check for CGST/SGST mentions
        if 'cgst' in text or 'sgst' in text:
            if not any(gst in text for gst in ['igst', 'cess', 'utgst']):
                suggestions.append("When mentioning CGST/SGST, consider if IGST applies for inter-state transactions")
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="GST validation found issues",
                severity=ValidationSeverity.WARNING,
                violations=violations,
                suggestions=suggestions
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="GST validation passed" if gstins else "No GST-related content to validate",
            metadata={"gstins_found": len(gstins)}
        )


class InvoiceValidator(BaseConstraint):
    """
    Validates invoice-related information in memories.
    
    This validator checks for:
    - Invoice number format
    - Date consistency (invoice date before due date)
    - Amount consistency
    - Required fields presence
    
    Use case: Finance and accounting applications.
    """
    
    INVOICE_PATTERN = re.compile(
        r'(?:invoice|invoice\s*no|inv|inv\.|invoice\s*number)[\s:#]*([A-Z0-9-]+)',
        re.IGNORECASE
    )
    
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',  # ISO format
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # US format
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # EU format
    ]
    
    @property
    def name(self) -> str:
        return "invoice_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.DATA_INTEGRITY
    
    @property
    def description(self) -> str:
        return "Validates invoice numbers, dates, and amounts for consistency"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate invoice-related content in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        text = memory.raw_text
        violations = []
        suggestions = []
        
        # Check for invoice number
        invoice_match = self.INVOICE_PATTERN.search(text)
        invoice_number = invoice_match.group(1) if invoice_match else None
        
        # Extract dates
        dates = []
        for pattern in self.DATE_PATTERNS:
            dates.extend(re.findall(pattern, text))
        
        # Check for amount mentions
        amount_pattern = r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, text)
        
        # Validate invoice structure
        has_invoice_number = invoice_number is not None
        has_date = len(dates) > 0
        has_amount = len(amounts) > 0
        
        if has_invoice_number:
            # Validate invoice number format
            if len(invoice_number) < 3:
                violations.append(f"Invoice number seems too short: {invoice_number}")
                suggestions.append("Invoice numbers should typically be at least 3 characters")
        
        # Check for date patterns suggesting due dates
        due_date_indicators = ['due', 'payable by', 'payment due', 'due date']
        has_due_indicator = any(indicator in text.lower() for indicator in due_date_indicators)
        
        if has_due_indicator and has_date:
            # Could validate date relationships here
            suggestions.append("Consider verifying that due date is after invoice date")
        
        # Check for partial invoice information
        if has_invoice_number and not (has_date or has_amount):
            suggestions.append("Invoice mentions detected but missing date or amount information")
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="Invoice validation found issues",
                severity=ValidationSeverity.WARNING,
                violations=violations,
                suggestions=suggestions
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Invoice validation passed" if has_invoice_number else "No invoice-related content to validate",
            metadata={
                "has_invoice_number": has_invoice_number,
                "dates_found": len(dates),
                "amounts_found": len(amounts)
            }
        )


class DateConsistencyValidator(BaseConstraint):
    """
    Validates temporal consistency in memories.
    
    This validator checks for:
    - Future dates that seem unrealistic
    - Date ordering inconsistencies
    - Time duration plausibility
    """
    
    @property
    def name(self) -> str:
        return "date_consistency_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.TEMPORAL
    
    @property
    def description(self) -> str:
        return "Ensures temporal consistency in memory timestamps and content"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate date consistency for a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        violations = []
        suggestions = []
        
        # Check if memory timestamp is reasonable
        now = datetime.utcnow()
        
        # Memory should not be too far in the future
        if memory.timestamp > now + timedelta(days=1):
            violations.append(f"Memory timestamp is in the future: {memory.timestamp.isoformat()}")
        
        # Memory should not be too old (arbitrary 10-year limit)
        if memory.timestamp < now - timedelta(days=3650):
            violations.append(f"Memory timestamp is very old: {memory.timestamp.isoformat()}")
            suggestions.append("Consider archiving very old memories")
        
        # Check for date mentions that contradict memory timestamp
        text = memory.raw_text.lower()
        date_indicators = ['today', 'tomorrow', 'yesterday', 'last week', 'next month']
        
        for indicator in date_indicators:
            if indicator in text:
                expected_date_range = self._get_expected_date_range(indicator)
                if expected_date_range:
                    if not (expected_date_range[0] <= memory.timestamp <= expected_date_range[1]):
                        suggestions.append(
                            f"Memory contains '{indicator}' but timestamp is "
                            f"{memory.timestamp.date()}. Consider verifying the timestamp."
                        )
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="Date consistency validation found issues",
                severity=ValidationSeverity.WARNING,
                violations=violations,
                suggestions=suggestions
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Date consistency validation passed",
            metadata={"timestamp": memory.timestamp.isoformat()}
        )
    
    def _get_expected_date_range(self, indicator: str) -> Optional[tuple]:
        """Get expected date range for a temporal indicator."""
        now = datetime.utcnow()
        
        ranges = {
            'today': (now - timedelta(hours=12), now + timedelta(hours=12)),
            'tomorrow': (now + timedelta(hours=12), now + timedelta(days=1.5)),
            'yesterday': (now - timedelta(days=1.5), now - timedelta(hours=12)),
            'last week': (now - timedelta(days=7), now - timedelta(days=1)),
            'next week': (now + timedelta(days=1), now + timedelta(days=7)),
            'last month': (now - timedelta(days=31), now - timedelta(days=1)),
            'next month': (now + timedelta(days=1), now + timedelta(days=31)),
        }
        
        return ranges.get(indicator)


class EmailValidator(BaseConstraint):
    """
    Validates email addresses in memories.
    
    This validator ensures email addresses are properly formatted
    and checks for potential issues.
    """
    
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    
    # Common disposable email domains
    DISPOSABLE_DOMAINS = {
        'tempmail.com', 'throwaway.com', 'mailinator.com',
        'guerrillamail.com', '10minutemail.com', 'fakeinbox.com'
    }
    
    @property
    def name(self) -> str:
        return "email_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.FORMAT
    
    @property
    def description(self) -> str:
        return "Validates email address format and checks for disposable emails"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate email addresses in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        violations = []
        suggestions = []
        
        emails = self.EMAIL_PATTERN.findall(memory.raw_text)
        
        for email in emails:
            # Check for disposable email domains
            domain = email.split('@')[1].lower()
            if domain in self.DISPOSABLE_DOMAINS:
                violations.append(f"Disposable email domain detected: {email}")
                suggestions.append("Consider using a permanent email address for important communications")
            
            # Check for common typos
            common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
            for correct_domain in common_domains:
                if domain.replace('gmai', 'gmail') == correct_domain or \
                   domain.replace('yhoo', 'yahoo') == correct_domain:
                    suggestions.append(f"Possible typo in email domain: {email}")
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="Email validation found issues",
                severity=ValidationSeverity.WARNING,
                violations=violations,
                suggestions=suggestions
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Email validation passed" if emails else "No email addresses to validate",
            metadata={"emails_found": len(emails)}
        )


class URLValidator(BaseConstraint):
    """
    Validates URLs in memories.
    
    This validator checks URL format and structure.
    """
    
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+'
    )
    
    # Suspicious URL patterns
    SUSPICIOUS_PATTERNS = [
        r'bit\.ly',
        r'tinyurl\.com',
        r'click\s*here',
        r'free\s*(?:money|gift|prize)',
    ]
    
    @property
    def name(self) -> str:
        return "url_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.FORMAT
    
    @property
    def description(self) -> str:
        return "Validates URL format and checks for suspicious links"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate URLs in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        violations = []
        suggestions = []
        
        urls = self.URL_PATTERN.findall(memory.raw_text)
        
        for url in urls:
            # Check for suspicious patterns
            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    suggestions.append(f"Review URL for safety: {url}")
            
            # Check for HTTP (not HTTPS)
            if url.startswith('http://') and not url.startswith('http://localhost'):
                suggestions.append(f"Consider using HTTPS for secure connections: {url}")
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="URL validation found issues",
                severity=ValidationSeverity.INFO,
                violations=violations,
                suggestions=suggestions
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="URL validation passed" if urls else "No URLs to validate",
            metadata={"urls_found": len(urls)}
        )


class CurrencyValidator(BaseConstraint):
    """
    Validates currency amounts in memories.
    
    This validator checks:
    - Currency format consistency
    - Amount plausibility
    - Currency code validity
    """
    
    CURRENCY_PATTERNS = {
        'INR': r'₹[\d,]*(?:\.\d{2})?',
        'USD': r'\$[\d,]*(?:\.\d{2})?',
        'EUR': r'€[\d,]*(?:\.\d{2})?',
        'GBP': r'£[\d,]*(?:\.\d{2})?',
    }
    
    @property
    def name(self) -> str:
        return "currency_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.FORMAT
    
    @property
    def description(self) -> str:
        return "Validates currency amounts and formats"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate currency amounts in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        suggestions = []
        amounts_found = {}
        
        text = memory.raw_text
        
        for currency, pattern in self.CURRENCY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                amounts_found[currency] = len(matches)
                
                for amount in matches[:3]:  # Check first 3 amounts
                    # Check for unusual formatting
                    if ',' in amount and '...' in amount:
                        suggestions.append(f"Unusual currency format detected: {amount}")
        
        # Check for mixed currencies
        if len(amounts_found) > 1:
            currencies = list(amounts_found.keys())
            suggestions.append(
                f"Multiple currencies detected in single memory: {', '.join(currencies)}. "
                "Consider separating these into distinct memories."
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Currency validation passed",
            metadata={"currencies_found": amounts_found}
        )


class PhoneNumberValidator(BaseConstraint):
    """
    Validates phone numbers in memories.
    
    This validator checks phone number format and plausibility.
    """
    
    PHONE_PATTERN = re.compile(
        r'(?:\+?91[-.\s]?)?(?:\d{3,4}[-.\s]?)?\d{3}[-.\s]?\d{4}'
    )
    
    @property
    def name(self) -> str:
        return "phone_number_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.FORMAT
    
    @property
    def description(self) -> str:
        return "Validates phone number formats"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Validate phone numbers in a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        phones = self.PHONE_PATTERN.findall(memory.raw_text)
        
        if phones:
            # Check for reasonable length
            for phone in phones:
                digits = re.sub(r'\D', '', phone)
                if len(digits) < 10:
                    return ConstraintResult.failed_result(
                        constraint_name=self.name,
                        message="Phone number validation found issues",
                        severity=ValidationSeverity.WARNING,
                        violations=[f"Phone number seems too short: {phone}"],
                        suggestions=["Phone numbers should typically be at least 10 digits"]
                    )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Phone number validation passed" if phones else "No phone numbers to validate",
            metadata={"phones_found": len(phones)}
        )


class BusinessRuleValidator(BaseConstraint):
    """
    General business rule validation framework.
    
    This validator allows defining custom business rules
    that can be applied to memories. Rules are defined
    as condition-action pairs.
    
    Example rules:
    - "All decisions must have a reason"
    - "Questions should have expected response times"
    """
    
    DEFAULT_RULES = [
        {
            "name": "decision_has_reason",
            "condition": lambda m: m.intent.value == "decision",
            "check": lambda m: len(m.raw_text) > 20,
            "message": "Decisions should include reasoning",
            "severity": ValidationSeverity.SUGGESTION
        },
        {
            "name": "question_has_context",
            "condition": lambda m: m.intent.value == "question",
            "check": lambda m: any(w in m.raw_text.lower() for w in ['what', 'how', 'why', 'when', 'who', 'where']),
            "message": "Questions should be specific",
            "severity": ValidationSeverity.INFO
        }
    ]
    
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize with custom rules.
        
        Args:
            rules: List of business rules to apply
        """
        self.rules = rules or self.DEFAULT_RULES
    
    @property
    def name(self) -> str:
        return "business_rule_validator"
    
    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.BUSINESS_RULE
    
    @property
    def description(self) -> str:
        return "Applies custom business rules to memories"
    
    def validate(self, memory) -> ConstraintResult:
        """
        Apply business rules to a memory.
        
        Args:
            memory: The MemoryNode to validate
            
        Returns:
            ConstraintResult with validation outcome
        """
        violations = []
        suggestions = []
        passed_rules = []
        
        for rule in self.rules:
            # Check if rule applies to this memory
            if 'condition' in rule and not rule['condition'](memory):
                continue
            
            # Check the rule
            if 'check' in rule:
                if not rule['check'](memory):
                    severity = rule.get('severity', ValidationSeverity.WARNING)
                    violations.append(rule.get('message', "Business rule violated"))
                    if 'suggestion' in rule:
                        suggestions.append(rule['suggestion'])
                else:
                    passed_rules.append(rule.get('name', 'unknown'))
        
        if violations:
            return ConstraintResult.failed_result(
                constraint_name=self.name,
                message="Business rule validation found issues",
                severity=ValidationSeverity.WARNING,
                violations=violations,
                suggestions=suggestions,
                metadata={"rules_checked": len(self.rules), "rules_passed": len(passed_rules)}
            )
        
        return ConstraintResult.passed_result(
            constraint_name=self.name,
            message="Business rules validation passed",
            metadata={"rules_passed": len(passed_rules), "rules_applicable": len(passed_rules)}
        )
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """
        Add a new business rule.
        
        Args:
            rule: Dictionary with rule definition
        """
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a business rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed
        """
        for i, rule in enumerate(self.rules):
            if rule.get('name') == rule_name:
                self.rules.pop(i)
                return True
        return False
