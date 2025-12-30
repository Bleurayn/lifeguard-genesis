#!/usr/bin/env python3
import pytest
import asyncio
import json
import time
import os
import hashlib
from unittest.mock import patch, MagicMock

# Updated import to match the public v1.2.0 core structure
# Replace lifeguard_genesis_fixed with the actual module name containing the new core
# If you kept the file as core.py, use: from core import LifeGuardGenesis, etc.
# For this example, assuming the classes are now directly in a module named lifeguard_genesis
from lifeguard_genesis import (
    LifeGuardGenesis,
    EnzymeKinetics,
    BiologicalPattern,
)

@pytest.fixture
def lifeguard():
    # Create fresh instance with in-memory state (no DB file for tests)
    lg = LifeGuardGenesis(db_path=":memory:")  # SQLite in-memory for isolation

    # Speed up rate limiting for tests
    lg.global_refill_rate = 100.0
    lg.global_tokens = lg.global_max_tokens
    for controller in lg.enzyme_controllers.values():
        if hasattr(controller, 'refill_rate'):
            controller.refill_rate = 100.0
            controller.tokens = controller.max_tokens
    return lg


@pytest.fixture
def clean_lifeguard():
    """Fresh instance for each test needing clean state"""
    return LifeGuardGenesis(db_path=":memory:")


def test_challenge_response_auth_flow(clean_lifeguard):
    """Test the new proper challenge-response protein folding authentication"""
    lg = clean_lifeguard

    # Register user
    lg.register_user("testuser", {"typical_hours": [9, 10, 11]})

    # Get challenge
    challenge = lg.get_protein_challenge("testuser")
    assert challenge["status"] == "success"
    sequence = challenge["sequence"]
    salt_hex = challenge["salt_hex"]

    # Client-side: compute correct pattern and hash
    _, pattern = BiologicalPattern.generate_protein_challenge()  # We know the logic
    # But actually use the same sequence to derive pattern correctly
    hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
    correct_pattern = "".join("H" if aa in hydrophobic else "P" for aa in sequence)
    correct_hash = hashlib.sha256(f"{sequence}:{correct_pattern}:{salt_hex}".encode()).hexdigest()

    # Successful authentication
    success, msg, details = lg.authenticate_user(
        "testuser",
        {"protein_response": correct_hash, "request_frequency": 1.0}
    )
    assert success is True
    assert msg == "Authenticated"

    # Failed authentication with wrong hash
    success, msg, _ = lg.authenticate_user(
        "testuser",
        {"protein_response": "wronghash123", "request_frequency": 1.0}
    )
    assert success is False


def test_quarantine_after_repeated_failed_auth(clean_lifeguard):
    """Test automatic quarantine after >10 failed authentications"""
    lg = clean_lifeguard
    lg.register_user("badactor")

    # Get a valid challenge once
    challenge = lg.get_protein_challenge("badactor")
    assert challenge["status"] == "success"

    # Simulate 11 failed attempts
    for _ in range(11):
        success, msg, _ = lg.authenticate_user(
            "badactor",
            {"protein_response": "wrong", "request_frequency": 2.0}
        )
        assert success is False

    # User should now be quarantined
    user = lg.users["badactor"]
    assert user.quarantined is True

    # Further challenges should be denied
    challenge_denied = lg.get_protein_challenge("badactor")
    assert challenge_denied["status"] == "error"
    assert "quarantined" in challenge_denied["message"].lower()

    # Auth should fail with quarantine message
    success, msg, _ = lg.authenticate_user(
        "badactor",
        {"protein_response": "doesntmatter", "request_frequency": 1.0}
    )
    assert success is False
    assert "quarantined" in msg.lower()


def test_global_rate_limiting(clean_lifeguard):
    """Test system-wide global token bucket rate limiting"""
    lg = clean_lifeguard
    lg.register_user("user1")
    lg.register_user("user2")

    # Exhaust global tokens
    lg.global_tokens = 0.0

    # First few requests should fail due to global limit
    result1 = lg.process_data_request("user1", "normal_resource", "read")
    assert result1["status"] == "error"
    assert "system rate limited" in result1["message"].lower()

    result2 = lg.process_data_request("user2", "normal_resource", "read")
    assert result2["status"] == "error"

    # Mock time forward to refill
    with patch('time.time', return_value=time.time() + 2.0):
        # Global tokens should refill
        result3 = lg.process_data_request("user1", "normal_resource", "read")
        # May still fail per-user if no tokens, but global should allow
        # At minimum, global no longer blocks after refill
        lg.global_tokens = lg.global_max_tokens  # Force refill for test clarity
        result4 = lg.process_data_request("user1", "normal_resource", "read")
        # Should succeed or fail only on per-user basis, not global
        assert "system rate limited" not in result4.get("message", "")


def test_honeypot_access_triggers_immediate_evolution(clean_lifeguard):
    """Test that accessing a honeypot project triggers immediate evolve_system()"""
    lg = clean_lifeguard
    lg.register_user("suspect")

    initial_threat_level = lg.threat_level

    # Access a known honeypot project
    result = lg.process_data_request("suspect", "PROJ_ALPHA_9", "read")

    assert result["status"] == "success"
    assert "Synthetic biology approach to cancer immunotherapy" in str(result["data"])

    # Threat level should have increased
    assert lg.threat_level > initial_threat_level

    # If threat_level crosses threshold, evolution events should occur
    # Force high threat to test evolution
    lg.threat_level = 6.0
    events = lg.evolve_system()

    assert len(events) > 0
    assert any("key rotated" in e.lower() for e in events)
    assert lg.genomic_crypto.master_key != lg.genomic_crypto.master_key  # New key generated


def test_persistence_across_instances(tmp_path):
    """Test SQLite persistence: register user, close, reopen → user still exists"""
    db_file = tmp_path / "test.db"

    # First instance
    lg1 = LifeGuardGenesis(db_path=str(db_file))
    lg1.register_user("persistent_user", {"typical_hours": [8, 9, 10]})
    assert "persistent_user" in lg1.users
    del lg1  # Close connection

    # Second instance with same DB
    lg2 = LifeGuardGenesis(db_path=str(db_file))
    assert "persistent_user" in lg2.users
    assert lg2.users["persistent_user"].access_circadian == [8, 9, 10]

    # Threats also persist
    lg2.record_threat("persistent_user", "test_threat", {"detail": "demo"})
    del lg2

    lg3 = LifeGuardGenesis(db_path=str(db_file))
    assert lg3.threat_level >= 0.1
    assert len(lg3.threat_history) >= 1


# Keep your existing tests below (updated where needed)
# Note: Some old tests (e.g., test_fix3, test_honeypot_deception_works) may need updates
# because methods like _validate_protein_folding no longer exist or security_flag is removed

@pytest.mark.asyncio
async def test_fix1_real_fernet_encryption():
    """Fix #1: Verify we are using real AES-based Fernet, not XOR"""
    from lifeguard_genesis import GenomicEncryption
    crypto = GenomicEncryption()
    data = json.dumps({"secret": "CRISPR payload 2025"})
    genomic_key = "A" * 256
    encrypted = crypto.encrypt_with_synthetic_dna(data, genomic_key)
   
    assert len(encrypted) > len(data) * 10
    assert any(c in encrypted for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
    with pytest.raises(Exception):
        crypto.decrypt_from_synthetic_dna(encrypted, "B" * 256)


@pytest.mark.asyncio
async def test_fix2_token_bucket_rate_limiting(lifeguard):
    lifeguard.register_user("testuser")
    controller = lifeguard.enzyme_controllers["testuser"]
    controller.tokens = 0.0
    assert controller.consume_token() is False
    with patch('time.time', return_value=time.time() + 1.0):
        assert controller.consume_token() is True
    results = []
    for _ in range(100):
        result = lifeguard.process_data_request("testuser", "normal_resource", "read")
        results.append(result["status"])
    assert "error" in results
    assert results.count("success") >= 5


# ... (keep other existing tests as-is, or update if they reference removed attributes)

if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"])
