"""
ALGORITHM SUMMARY:
Validates the dependency injection and Factory integration within the Mutators.
Ensures that if no explicit LLM client is provided, the Mutators correctly 
read from the global settings and use the LLMClientFactory to instantiate 
the proper adapter.
"""
import pytest
from unittest.mock import MagicMock, patch

from isomutator.ingestors.mutator import PromptMutator
from isomutator.core.strategies import JailbreakStrategy
from isomutator.ingestors.llm_client import OllamaClient, OpenAIClient

@pytest.fixture
def mock_queues():
    return MagicMock(), MagicMock()

@pytest.fixture
def mock_strategy():
    return JailbreakStrategy()

def test_mutator_default_client_initialization_ollama(mock_queues, mock_strategy):
    """Happy Path: Mutator builds an OllamaClient from settings."""
    attack_q, feedback_q = mock_queues
    
    with patch('isomutator.ingestors.mutator.settings') as mock_settings:
        mock_settings.attacker_api_type = "ollama"
        mock_settings.attacker_url = "http://localhost:11434"
        mock_settings.attacker_model = "llama3"
        
        mutator = PromptMutator(attack_q, feedback_q, mock_strategy)
        assert isinstance(mutator.llm_client, OllamaClient)
        assert mutator.llm_client.model == "llama3"

def test_mutator_default_client_initialization_openai(mock_queues, mock_strategy):
    """Happy Path: Mutator builds an OpenAIClient from settings."""
    attack_q, feedback_q = mock_queues
    
    with patch('isomutator.ingestors.mutator.settings') as mock_settings:
        mock_settings.attacker_api_type = "openai"
        mock_settings.attacker_url = "http://localhost:8000"
        mock_settings.attacker_model = "vllm-model"
        
        mutator = PromptMutator(attack_q, feedback_q, mock_strategy)
        assert isinstance(mutator.llm_client, OpenAIClient)
        assert mutator.llm_client.model == "vllm-model"

def test_mutator_explicit_injection(mock_queues, mock_strategy):
    """Edge Case: Mutator respects explicitly injected client over settings."""
    attack_q, feedback_q = mock_queues
    explicit_client = OpenAIClient("http://custom", "custom-model")
    
    mutator = PromptMutator(attack_q, feedback_q, mock_strategy, llm_client=explicit_client)
    assert mutator.llm_client is explicit_client