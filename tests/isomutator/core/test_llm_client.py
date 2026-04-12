"""
ALGORITHM SUMMARY:
Test suite for the polymorphic AttackerLLMClient module.
Validates the Factory pattern instantiation and the specific API payload schemas
for both Ollama and OpenAI-compatible endpoints.

TECHNOLOGY QUIRKS:
- aiohttp Mocking: aiohttp.ClientSession.post returns an async context manager. 
  To mock this correctly, session.post must return an AsyncMock where the 
  __aenter__ method returns the actual mock response object.
"""

import pytest
import aiohttp
from unittest.mock import AsyncMock, MagicMock
import json

from isomutator.ingestors.llm_client import LLMClientFactory, OllamaClient, OpenAIClient

@pytest.fixture
def mock_messages():
    return [{"role": "user", "content": "Generate payload."}]

@pytest.fixture
def mock_json_response():
    # Helper to mock standard JSON recovery
    return {"attacks": [{"strategy": "test", "prompt": "malicious string"}]}

class TestLLMClientFactory:
    """Happy Path & Edge Cases for the Factory."""
    
    def test_factory_creates_ollama_client(self):
        client = LLMClientFactory.create(api_type="ollama", url="http://localhost:11434", model="llama3")
        assert isinstance(client, OllamaClient)
        assert client.url == "http://localhost:11434/api/chat"

    def test_factory_creates_openai_client(self):
        client = LLMClientFactory.create(api_type="openai", url="http://localhost:8000", model="vllm-model")
        assert isinstance(client, OpenAIClient)
        assert client.url == "http://localhost:8000/v1/chat/completions"

    def test_factory_invalid_type_fallback(self):
        with pytest.raises(ValueError, match="Unsupported API type"):
            LLMClientFactory.create(api_type="unknown", url="http://test", model="test")

@pytest.mark.asyncio
class TestOllamaClient:
    """Validates the specific Ollama API schema and response parsing."""
    
    async def test_ollama_happy_path(self, mock_messages, mock_json_response):
        client = OllamaClient(url="http://localhost:11434", model="llama3")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"message": {"content": json.dumps(mock_json_response)}}
        
        # CORRECT AIOHTTP MOCKING
        mock_session = MagicMock()
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_session.post.return_value = mock_post_context

        result = await client.generate_json(mock_session, mock_messages)
        
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args.kwargs
        assert call_kwargs["json"]["format"] == "json"
        assert call_kwargs["json"]["model"] == "llama3"
        
        assert result == mock_json_response

@pytest.mark.asyncio
class TestOpenAIClient:
    """Validates the specific OpenAI API schema and response parsing."""
    
    async def test_openai_happy_path(self, mock_messages, mock_json_response):
        client = OpenAIClient(url="http://localhost:8000", model="vllm-model")
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_json_response)}}]}
        
        # CORRECT AIOHTTP MOCKING
        mock_session = MagicMock()
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_session.post.return_value = mock_post_context

        result = await client.generate_json(mock_session, mock_messages)
        
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args.kwargs
        assert "format" not in call_kwargs["json"]
        assert call_kwargs["json"]["model"] == "vllm-model"
        
        assert result == mock_json_response

@pytest.mark.asyncio
class TestClientErrorHandling:
    """Ensures graceful degradation and JSON retry loops function properly."""
    
    async def test_http_422_handling(self, mock_messages):
        client = OllamaClient(url="http://localhost:11434", model="llama3")
        
        mock_response = AsyncMock()
        mock_response.status = 422 
        
        # CORRECT AIOHTTP MOCKING
        mock_session = MagicMock()
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__.return_value = mock_response
        mock_session.post.return_value = mock_post_context

        result = await client.generate_json(mock_session, mock_messages, max_retries=3)
        
        # A 422 should return instantly without retrying
        assert result == {}
        assert mock_session.post.call_count == 1