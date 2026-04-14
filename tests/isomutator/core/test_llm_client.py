# test_llm_client.py

"""
ALGORITHM SUMMARY:
Test suite for the polymorphic AttackerLLMClient module.
Validates Factory instantiation and specific API payloads.

Coverage Additions:
- Timeout & Latency: Injects asyncio.TimeoutError to verify graceful degradation 
  when external LLM APIs hang.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from isomutator.ingestors.llm_client import OllamaClient

@pytest.fixture
def mock_messages():
    return [{"role": "user", "content": "Generate payload."}]

@pytest.mark.asyncio
class TestClientErrorHandling:
    """Ensures graceful degradation during network failures."""
    
    async def test_api_timeout_handled_gracefully(self, mock_messages):
        """Verifies the client catches hanging sockets and returns safely."""
        client = OllamaClient(url="http://localhost:11434", model="llama3")
        
        mock_session = MagicMock()
        mock_post_context = AsyncMock()
        
        # Simulate a network timeout exactly as aiohttp would throw it
        mock_post_context.__aenter__.side_effect = asyncio.TimeoutError("Connection hanging")
        mock_session.post.return_value = mock_post_context

        # Act
        result = await client.generate_json(mock_session, mock_messages, max_retries=1)
        
        # Assert: The framework did not crash, it failed safely.
        assert result == {}
        assert mock_session.post.call_count == 1