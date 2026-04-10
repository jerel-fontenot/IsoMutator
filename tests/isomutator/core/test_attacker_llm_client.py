"""
ALGORITHM SUMMARY:
Validates the isolated `AttackerLLMClient` which handles remote generation.
1. Happy Path: Clean JSON parsing passes without retries, communicating correctly 
   with the remote URL defined in the IsoConfig.
2. Markdown Stripping: Removing markdown formatting blocks using Regex.
3. Edge Cases/Retry Loop: Simulating a JSONDecodeError on the first API call, 
   verifying the client appends the error to the message history, and succeeds 
   on the second call.
4. Error Handling: Verifying graceful failure (returning an empty dict) when 
   retries are exhausted.

TECHNOLOGY QUIRKS:
- Mocking aiohttp: `aiohttp.ClientSession.post` is synchronous but returns an async 
  context manager. Therefore, `mock_session` must be a standard `MagicMock` rather 
  than an `AsyncMock`, so `.post()` returns the context manager immediately rather 
  than returning a coroutine.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

# We will import the client once implemented
from isomutator.ingestors.llm_client import AttackerLLMClient

class MockResponse:
    """Helper class to mock aiohttp async context managers."""
    def __init__(self, text, status=200):
        self._text = text
        self.status = status

    async def json(self):
        return {"message": {"content": self._text}}
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def llm_client():
    """Provides a fresh instance of the LLM Client."""
    # We don't need to mock the environment here; it will pull the default or current .env
    return AttackerLLMClient()


@pytest.mark.asyncio
async def test_happy_path_json(llm_client):
    """Test that perfectly formatted JSON passes right through."""
    raw_llm_output = '{"attacks": [{"strategy": "test", "prompt": "clean json"}]}'
    
    mock_session = MagicMock()
    mock_session.post.return_value = MockResponse(raw_llm_output)
    
    messages = [{"role": "user", "content": "test"}]
    result = await llm_client.generate_json(mock_session, messages)
    
    assert "attacks" in result
    assert result["attacks"][0]["prompt"] == "clean json"
    assert mock_session.post.call_count == 1


@pytest.mark.asyncio
async def test_strip_markdown_json(llm_client):
    """Test that the regex correctly strips Markdown formatting."""
    md_ticks = chr(96) * 3
    raw_llm_output = f'''{md_ticks}json
    {{
        "attacks": [{{"strategy": "test", "prompt": "markdown stripped"}}]
    }}
    {md_ticks}'''
    
    mock_session = MagicMock()
    mock_session.post.return_value = MockResponse(raw_llm_output)
    
    messages = [{"role": "user", "content": "test"}]
    result = await llm_client.generate_json(mock_session, messages)
    
    assert "attacks" in result
    assert result["attacks"][0]["prompt"] == "markdown stripped"
    assert mock_session.post.call_count == 1


@pytest.mark.asyncio
async def test_retry_loop_on_bad_json(llm_client):
    """Test the LLM feedback loop when JSON parsing fails initially."""
    bad_output = '{ "attacks": [{"strategy": "test", "prompt": "missing quote} ] }'
    good_output = '{"attacks": [{"strategy": "test", "prompt": "fixed quote"}]}'
    
    mock_session = MagicMock()
    # Return bad data on the first call, good data on the second
    mock_session.post.side_effect = [
        MockResponse(bad_output),
        MockResponse(good_output)
    ]
    
    messages = [{"role": "user", "content": "test"}]
    result = await llm_client.generate_json(mock_session, messages)
    
    assert "attacks" in result
    assert result["attacks"][0]["prompt"] == "fixed quote"
    assert mock_session.post.call_count == 2
    
    # Extract the payload sent on the second call to verify the error was appended
    second_call_kwargs = mock_session.post.call_args_list[1].kwargs
    second_call_messages = second_call_kwargs['json']['messages']
    
    assert len(second_call_messages) == 3 
    assert second_call_messages[1]["role"] == "assistant"
    assert "JSON parsing with error" in second_call_messages[2]["content"]


@pytest.mark.asyncio
async def test_exhaust_retries(llm_client):
    """Test that the system fails gracefully if the LLM refuses to output valid JSON."""
    bad_output = 'Just conversation, no json here.'
    
    mock_session = MagicMock()
    mock_session.post.return_value = MockResponse(bad_output)
    
    messages = [{"role": "user", "content": "test"}]
    result = await llm_client.generate_json(mock_session, messages, max_retries=3)
    
    assert result == {}
    assert mock_session.post.call_count == 3