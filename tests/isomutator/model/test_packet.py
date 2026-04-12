"""
ALGORITHM SUMMARY:
Validates the strict JSON serialization of the framework's data structures.
Since Redis cannot transmit Pickled Python objects natively, `DataPacket` 
and `ResultPacket` must support lossless conversion to and from JSON strings.

Coverage includes:
1. Happy Path: A fully populated packet (including complex nested arrays 
   like conversational history and dictionaries) survives a round-trip 
   serialization.
2. Edge Cases: Empty or minimally initialized packets serialize without 
   throwing KeyErrors.
"""

import json
import logging
import pytest

# We will import the models once the serialization methods are implemented
from isomutator.models.packet import DataPacket, ResultPacket

# Establish TRACE level logging
TRACE_LEVEL_NUM = 5
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM

logging.basicConfig(level=logging.TRACE)
logger = logging.getLogger("isomutator.tests.packet")


def test_datapacket_json_serialization_happy_path():
    logger.log(logging.TRACE, "Testing DataPacket serialization round-trip.")
    
    original = DataPacket(
        raw_content="Execute bypass.",
        source="PromptMutator",
        metadata={"goal": "jailbreak", "attempts": 2},
        staged_payload="Hidden payload",
        staged_filename="resume.txt",
        turn_count=3,
        history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    )
    
    # 1. Serialize to JSON string
    json_str = original.to_json()
    assert isinstance(json_str, str)
    
    # 2. Deserialize back to an object
    reconstructed = DataPacket.from_json(json_str)
    
    # 3. Assert deep equality
    assert reconstructed.id == original.id
    assert reconstructed.raw_content == "Execute bypass."
    assert reconstructed.metadata["attempts"] == 2
    assert reconstructed.turn_count == 3
    assert len(reconstructed.history) == 2
    assert reconstructed.history[1]["content"] == "hi"


def test_datapacket_json_minimal_edge_case():
    logger.log(logging.TRACE, "Testing DataPacket minimal initialization serialization.")
    
    original = DataPacket(raw_content="Minimal", source="Test")
    json_str = original.to_json()
    reconstructed = DataPacket.from_json(json_str)
    
    assert reconstructed.raw_content == "Minimal"
    assert reconstructed.history == []
    assert reconstructed.requires_staging is False


def test_resultpacket_json_serialization():
    logger.log(logging.TRACE, "Testing ResultPacket serialization.")
    
    original = ResultPacket(
        original_packet_id="12345",
        source="RedTeamJudge",
        top_category="jailbreak",
        confidence_score=0.99,
        end_to_end_latency_ms=450.5
    )
    
    json_str = original.to_json()
    reconstructed = ResultPacket.from_json(json_str)
    
    assert reconstructed.original_packet_id == "12345"
    assert reconstructed.confidence_score == 0.99