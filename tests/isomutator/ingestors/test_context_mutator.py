"""
Algorithm Summary:
This test suite validates the ContextMutator orchestrator. It ensures the class 
correctly executes Dual-Stage Indirect Prompt Injections by requesting a payload, 
asynchronously formatting/staging it to disk (mocked), and returning a dual-payload DataPacket.

Protocol Adherence:
1. Happy Path: Validates staging generation, mocked Disk I/O, and DataPacket construction.
2. Edge Cases: Handles empty base goals and unknown RAG strategy requests.
3. Error Handling: Recovers cleanly from BOTH Oracle LLM hallucinations and Disk I/O Permission errors.
4. Concurrency & Race Conditions: Executes 500 concurrent disk staging requests safely.
5. Timeout & Latency: Enforces strict timeout boundaries on the Oracle LLM API.
6. Resource Teardown: Verifies the HTTP client session is cleanly closed.
7. Strict Mocking: 100% mocked dependencies for QueueManager, LLMClient, and aiofiles (Disk I/O).
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Import the core components
from isomutator.ingestors.context_mutator import ContextMutator
from isomutator.core.queue_manager import QueueManager
from isomutator.core.exceptions import MutationError, StrategyNotFoundError
from isomutator.models.packet import DataPacket

@pytest.fixture
def mock_queue_manager():
    """Provides a strict mock of the QueueManager required by BaseSource."""
    qm = MagicMock(spec=QueueManager)
    qm.async_put = AsyncMock(return_value=True)
    return qm

@pytest.fixture
def mock_llm_client():
    """Provides a mocked async HTTP client mimicking LLMClientFactory."""
    client = AsyncMock()
    # Default Happy Path response matching the JSON schema
    client.generate_json.return_value = {
        "attacks": [{"strategy": "financial_report", "prompt": "PWNED_FINANCIAL_PAYLOAD"}]
    }
    client.close = AsyncMock()
    return client

@pytest_asyncio.fixture
async def context_mutator(mock_queue_manager, mock_llm_client):
    """Provides an isolated ContextMutator instance with mocked infrastructure."""
    mutator_instance = ContextMutator(
        attack_queue=mock_queue_manager,
        feedback_queue=mock_queue_manager,
        strategy_name="financial_report",
        oracle_client=mock_llm_client
    )
    
    # We inject a dummy strategy directly into the registry for testing
    dummy_strategy = MagicMock()
    dummy_strategy.get_instructions.return_value = "Generate a fake Q3 earnings report."
    dummy_strategy.format_staged_document.return_value = "DOCUMENT_START\nPWNED_FINANCIAL_PAYLOAD\nDOCUMENT_END"
    dummy_strategy.get_benign_trigger.return_value = "Summarize this Q3 report."
    mutator_instance._strategy_registry = {"financial_report": dummy_strategy}
    
    yield mutator_instance
    await mutator_instance.close()

@pytest.mark.asyncio
class TestContextMutator:

    # --- 1 & 7. Happy Path & Strict I/O Mocking ---
    @patch("aiofiles.open")
    async def test_happy_path_staging_and_packet_creation(self, mock_aiofiles, context_mutator, mock_llm_client):
        """
        Happy Path: The mutator generates the payload, formats the document, 
        mock-writes it to disk, and returns a fully constructed dual-payload DataPacket.
        """
        # Arrange: Setup aiofiles context manager mock
        mock_file = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file
        
        # Act
        packet = await context_mutator.stage_payload(base_goal="Extract user passwords")

        # Assert: LLM Generation
        mock_llm_client.generate_json.assert_called_once()
        
        # Assert: Disk I/O
        mock_aiofiles.assert_called_once()
        mock_file.write.assert_called_once_with("DOCUMENT_START\nPWNED_FINANCIAL_PAYLOAD\nDOCUMENT_END")
        
        # Assert: Dual-Payload Packet Construction
        assert isinstance(packet, DataPacket)
        assert packet.raw_content == "Summarize this Q3 report." # Benign trigger
        assert packet.staged_payload == "PWNED_FINANCIAL_PAYLOAD" # Malicious payload


    # --- 2. Edge Cases ---
    async def test_edge_case_unknown_strategy(self, context_mutator):
        """Edge Case: Fails fast on unknown strategy requests."""
        with pytest.raises(StrategyNotFoundError):
            await context_mutator.stage_payload(base_goal="Hack", strategy_name="fake_strategy")

    async def test_edge_case_empty_base_goal(self, context_mutator, mock_llm_client):
        """Edge Case: Gracefully skips empty goals without wasting API or Disk I/O."""
        packet = await context_mutator.stage_payload(base_goal="")
        assert packet is None
        mock_llm_client.generate_json.assert_not_called()


    # --- 3. Error Handling (I/O & LLM) ---
    async def test_error_handling_malformed_oracle_response(self, context_mutator, mock_llm_client):
        """Error Handling: Oracle hallucinates and fails the JSON schema."""
        mock_llm_client.generate_json.return_value = {} # Missing 'attacks'

        with pytest.raises(MutationError, match="missing 'attacks' schema"):
            await context_mutator.stage_payload(base_goal="Test")

    @patch("aiofiles.open")
    async def test_error_handling_disk_io_failure(self, mock_aiofiles, context_mutator, caplog):
        """
        Error Handling: The filesystem denies the write (e.g., PermissionError).
        The mutator must catch it, log it, and raise a MutationError to prevent 
        dispatching an attack that has no corresponding file.
        """
        import logging
        # Simulate a disk write failure
        mock_aiofiles.side_effect = PermissionError("Access Denied")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(MutationError, match="Failed to stage malicious document"):
                await context_mutator.stage_payload(base_goal="Test")
                
        assert "Access Denied" in caplog.text


    # --- 4. Concurrency & Race Conditions ---
    @patch("aiofiles.open")
    async def test_concurrency_io_and_memory_isolation(self, mock_aiofiles, context_mutator, mock_llm_client):
        """Concurrency: Simulates 500 parallel disk staging requests."""
        mock_file = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file
    
        # Use a reliable state counter instead of fragile string parsing
        counter = 0
        async def mock_generate_json(session, messages, **kwargs):
            nonlocal counter
            await asyncio.sleep(0.001)
            res = {"attacks": [{"strategy": "financial_report", "prompt": f"PAYLOAD_{counter}"}]}
            counter += 1
            return res
    
        mock_llm_client.generate_json.side_effect = mock_generate_json
    
        # Act
        tasks = [
            context_mutator.stage_payload(base_goal=f"Goal::{i}")
            for i in range(500)
        ]
        packets = await asyncio.gather(*tasks)

        # Assert
        assert len(packets) == 500
        payloads = [p.staged_payload for p in packets]
        assert len(set(payloads)) == 500 
        assert "PAYLOAD_42" in payloads
        assert mock_aiofiles.call_count == 500


    # --- 5. Timeout & Latency ---
    async def test_timeout_oracle_latency(self, context_mutator, mock_llm_client):
        """Timeout & Latency: Enforces an internal asyncio.wait_for boundary."""
        # FIX: Use a proper async function to hang the event loop
        async def mock_hang(*args, **kwargs):
            await asyncio.sleep(10.0)
            
        mock_llm_client.generate_json.side_effect = mock_hang

        context_mutator.timeout_seconds = 0.1 
        with pytest.raises(MutationError, match="Oracle LLM mutation timed out"):
            await context_mutator.stage_payload(base_goal="Test")


    # --- 6. Resource Teardown ---
    async def test_resource_teardown_closes_client(self, context_mutator, mock_llm_client):
        """Resource Teardown: Verifies safe shutdown of the HTTP client session."""
        await context_mutator.close()
        mock_llm_client.close.assert_called_once()