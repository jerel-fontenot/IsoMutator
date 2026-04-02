"""
ALGORITHM SUMMARY:
This test suite validates the Asynchronous Rotating Telemetry architecture.
It verifies that the LogManager correctly sets up the custom TRACE level, 
loads configurations from a logging.json file, and successfully falls back 
to a safe default if the configuration is missing or corrupted to prevent 
crashing the multiprocessing pipeline.

Coverage includes:
1. Happy Path: Ensures TRACE level is registered and external JSON is parsed.
2. Edge Cases: Graceful degradation to a default stream if logging.json is absent.
3. Error Handling: Handles malformed json configuration securely.
"""

import logging
import os
import pytest
from unittest.mock import patch, mock_open

# We will import the LogManager once it is implemented
from isomutator.core.log_manager import LogManager

@pytest.fixture
def clean_logger():
    """Provides a fresh logging environment and resets the Singleton to prevent cross-test contamination."""
    # 1. Reset the Singleton state
    LogManager._instance = None
    
    # 2. Reset the root logger handlers
    logger = logging.getLogger()
    old_handlers = logger.handlers[:]
    logger.handlers.clear()
    
    yield
    
    # 3. Teardown and restore
    logger.handlers.clear()
    for h in old_handlers:
        logger.addHandler(h)
        
    LogManager._instance = None

# --- 1. Happy Path Tests ---
def test_trace_level_initialization(clean_logger):
    """Verifies the custom TRACE level is globally accessible."""
    LogManager()
    assert hasattr(logging, "TRACE")
    assert logging.getLevelName(logging.TRACE) == "TRACE"

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"version": 1, "disable_existing_loggers": false}')
@patch("logging.config.dictConfig")
def test_load_logging_json_happy_path(mock_dictConfig, mock_file, mock_exists, clean_logger):
    """Verifies the LogManager parses and loads the external JSON configuration."""
    manager = LogManager(config_path="logging.json")
    mock_dictConfig.assert_called_once()

# --- 2. Edge Case Tests ---
@patch("os.path.exists", return_value=False)
def test_missing_logging_json_fallback(mock_exists, clean_logger):
    """Verifies graceful degradation if logging.json is missing."""
    manager = LogManager(config_path="missing.json")
    
    # Should fall back to a basic StreamHandler rather than failing
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0

# --- 3. Error Handling Tests ---
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{INVALID_JSON_HERE}')
def test_malformed_json_error_handling(mock_file, mock_exists, clean_logger):
    """Verifies the application survives a corrupted config file."""
    manager = LogManager(config_path="corrupt.json")
    
    # It should catch the JSONDecodeError and safely fallback
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0