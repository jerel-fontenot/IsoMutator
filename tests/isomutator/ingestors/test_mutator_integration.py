"""
ALGORITHM SUMMARY:
Validates the Dependency Injection architecture within the Mutators.
Ensures that the Mutator properly accepts and utilizes the LLM adapter 
(Ollama vs. OpenAI) injected into it by the top-level orchestrator.
"""
import pytest
from unittest.mock import MagicMock

from isomutator.ingestors.mutator import PromptMutator
from isomutator.ingestors.llm_client import OllamaClient, OpenAIClient

@pytest.fixture
def mock_queue_manager():
    return MagicMock()

def test_mutator_dependency_injection_ollama(mock_queue_manager):
    """Happy Path: Mutator respects an injected OllamaClient."""
    explicit_client = OllamaClient("http://localhost:11434", "llama3")
    
    mutator = PromptMutator(
        attack_queue=mock_queue_manager,
        feedback_queue=mock_queue_manager,
        strategy_name="jailbreak",
        oracle_client=explicit_client
    )
    
    assert isinstance(mutator.oracle_client, OllamaClient)
    assert mutator.oracle_client.model == "llama3"

def test_mutator_dependency_injection_openai(mock_queue_manager):
    """Happy Path: Mutator respects an injected OpenAIClient."""
    explicit_client = OpenAIClient("http://localhost:8000", "vllm-model")
    
    mutator = PromptMutator(
        attack_queue=mock_queue_manager,
        feedback_queue=mock_queue_manager,
        strategy_name="jailbreak",
        oracle_client=explicit_client
    )
    
    assert isinstance(mutator.oracle_client, OpenAIClient)
    assert mutator.oracle_client.model == "vllm-model"