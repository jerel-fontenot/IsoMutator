# test_config.py

"""
ALGORITHM SUMMARY:
This test suite validates the `IsoConfig` manager, ensuring it properly 
loads, sanitizes, and validates environment variables using Pydantic.

Coverage includes:
1. Happy Path: Verifies that the remote CorpRAG-Target URL and Attacker URLs 
   are correctly established, either by default or via environment overrides.
2. Edge Cases: Ensures URL trailing slashes are automatically stripped to 
   prevent double-slash routing errors (e.g., `http://host//api`).
3. Error Handling: Verifies that malformed URLs trigger a strict validation 
   error, failing fast before the networking layer attempts a connection.

TECHNOLOGY QUIRKS:
- Pydantic Settings Mocking: Because Pydantic reads from the OS environment 
  at instantiation, we use `unittest.mock.patch.dict` to inject mock `ISO_` 
  environment variables safely during test isolation.
"""

import logging
import pytest
from pydantic import ValidationError
from unittest.mock import patch

# We will import the config once updated
from isomutator.core.config import IsoConfig

# Establish TRACE level logging
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

logging.basicConfig(level=logging.TRACE)
logger = logging.getLogger("isomutator.tests.config")


# --- 1. Happy Path Tests ---
def test_config_default_remote_targets():
    logger.log(logging.TRACE, "Testing default remote URL initialization.")
    # Without environment variables, it should default to the defined remote servers
    config = IsoConfig()
    
    assert config.target_url == "http://192.9.159.125:8000"
    assert config.attacker_url == "http://192.9.159.125:11434"


@patch.dict('os.environ', {'ISO_TARGET_URL': 'http://10.0.0.5:9000'})
def test_config_environment_override():
    logger.log(logging.TRACE, "Testing environment variable override mapping.")
    # Pydantic should map ISO_TARGET_URL to the target_url property
    config = IsoConfig()
    assert config.target_url == "http://10.0.0.5:9000"


# --- 2. Edge Case Tests ---
@patch.dict('os.environ', {'ISO_TARGET_URL': 'http://192.9.159.125:8000/'})
def test_config_strips_trailing_slashes():
    logger.log(logging.TRACE, "Testing URL sanitization (trailing slashes).")
    config = IsoConfig()
    # The trailing slash must be stripped to prevent routing bugs downstream
    assert config.target_url == "http://192.9.159.125:8000"


# --- 3. Error Handling Tests ---
@patch.dict('os.environ', {'ISO_TARGET_URL': 'not_a_valid_url'})
def test_config_invalid_url_fails_fast():
    logger.log(logging.TRACE, "Testing strict URL validation.")
    # Pydantic's AnyHttpUrl type should raise a ValidationError on bad formats
    with pytest.raises(ValidationError):
        IsoConfig()