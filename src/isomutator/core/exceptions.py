"""
=============================================================================
IsoMutator Core: Exceptions
=============================================================================

Algorithm Summary:
Defines the custom exception hierarchy for the IsoMutator application. 
By utilizing custom exception classes, we ensure that error handling (like 
catching LLM timeouts or missing strategies) is highly specific. This prevents 
broad `except Exception:` blocks from accidentally swallowing critical system 
errors (like memory allocation failures or KeyboardInterrupts).
"""

class IsoMutatorError(Exception):
    """
    Base exception class for all custom IsoMutator errors.
    Allows top-level handlers to catch any engine-specific error safely.
    """
    pass

class MutationError(IsoMutatorError):
    """
    Raised when the core Intelligence engine fails to generate or stage a payload.
    Common triggers:
    - Oracle LLM asyncio timeouts
    - JSONDecodeErrors / Schema Validation failures
    - Disk I/O Permission errors during document staging
    """
    pass

class StrategyNotFoundError(IsoMutatorError):
    """
    Raised when the Factory registry is asked to load an attack strategy 
    that does not exist or has not been registered.
    """
    pass

class BrokerConnectionError(IsoMutatorError):
    """
    Raised when the multiprocessing queue manager or Redis broker becomes 
    unreachable during telemetry dispatch or teardown sequences.
    """
    pass

class ReportingError(Exception):
    """Raised when the ReportGenerator fails to parse telemetry or export data."""
    pass