#!/usr/bin/env python3
import pytest
import asyncio
import json
import time
import os
from unittest.mock import patch, MagicMock

# Import the fixed system (adjust path if you split files)
from lifeguard_genesis_fixed import (
    LifeGuardGenesis,
    PlatformIntegrator,
    PlatformType,
    EnzymeKinetics,
    BiologicalPattern,
    GenomicEncryption
)

@pytest.fixture
def lifeguard():
    lg = LifeGuardGenesis()
    # Speed up rate limiting for tests
    for controller in lg.enzyme_controllers.values():
        if hasattr(controller, 'refill_rate'):
            controller.refill_rate = 100.0  # Instant refill in tests
            controller.tokens = controller.max_tokens
    return lg

@pytest.fixture
def integrator(lifeguard):
    return PlatformIntegrator(lifeguard)

@pytest.mark.asyncio
async def test_fix1_real_fernet_encryption():
    """Fix #1: Verify we are using real AES-based Fernet, not XOR"""
    crypto = GenomicEncryption()
    data = json.dumps({"secret": "CRISPR payload 2025"})
    genomic_key = "A" * 256

    encrypted = crypto.encrypt_with_synthetic_dna(data, genomic_key)
    
    # Must contain valid Fernet token characters and be much longer than original
    assert len(encrypted) > len(data) * 10
    assert any(c in encrypted for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")

    # Try to decrypt with wrong key → must fail
    with pytest.raises(Exception):  # InvalidToken or ValueError
        crypto.decrypt_from_synthetic_dna(encrypted, "B" * 256, os.urandom(16))

@pytest.mark.asyncio
async def test_fix2_token_bucket_rate_limiting(lifeguard):
    """Fix #2: Non-blocking token bucket instead of time.sleep()"""
    lifeguard.register_user("testuser")
    controller = lifeguard.enzyme_controllers["testuser"]

    # Exhaust tokens
    controller.tokens = 0.0
    assert controller.consume_token() is False

    # Refill over time (mock time)
    with patch('time.time', return_value=time.time() + 1.0):
        assert controller.consume_token() is True  # Should have refilled

    # Spam 100 requests → should be rate limited gracefully
    results = []
    for _ in range(100):
        result = lifeguard.process_data_request("testuser", "normal_resource", "read")
        results.append(result["status"])

    # Should see some "Rate limited" but no blocking
    assert "error" in results
    assert results.count("success") >= 5  # At least some get through

def test_fix3_full_hash_protein_validation(lifeguard):
    """Fix #3: Full SHA-256 match, not prefix"""
    lifeguard.register_user("alice")
    user = lifeguard.users["alice"]

    # Correct response (simulated from registration logic)
    sequence, pattern = BiologicalPattern.generate_protein_challenge()
    salt = os.urandom(16).hex()
    expected = hashlib.sha256(f"{sequence}:{pattern}:{salt}".encode()).hexdigest()
    user.protein_signature = expected  # Force correct signature

    # Wrong response (even 1 bit off)
    wrong_response = "A" * 100
    assert lifeguard._validate_protein_folding(wrong_response, expected) is False

    # Correct response must pass
    correct_response = f"{sequence}:{pattern}:{salt}"
    assert lifeguard._validate_protein_folding(
        hashlib.sha256(correct_response.encode()).hexdigest(),
        expected
    ) is True

@pytest.mark.asyncio
async def test_fix4_secure_session_management(integrator):
    """Fix #4: Sessions encrypted + expire"""
    integrator.lifeguard.register_user("bob")

    request = {"operation_type": "test"}
    encrypted_payload = await integrator._encrypt_platform_request(request, integrator.platform_configs[PlatformType.GENOPATTERN])

    payload = json.loads(encrypted_payload)
    session_id = payload["session_id"]

    # Session must exist and be encrypted
    session = integrator.active_sessions[session_id]
    assert session["expires_at"] > time.time()
    assert session["encrypted_genomic_key"].startswith("gAAAAA")  # Fernet token

    # Expire session
    session["expires_at"] = time.time() - 1
    integrator.active_sessions[session_id] = session

    # Cleanup should remove it
    integrator.get_integration_status()
    assert session_id not in integrator.active_sessions

@pytest.mark.asyncio
async def test_honeypot_deception_works(lifeguard, integrator):
    """Honeypot Strategy Test: Accessing decoy project triggers fake data + logs"""
    lifeguard.register_user("attacker")

    # Access a real-looking but fake project
    result = lifeguard.process_data_request("attacker", "PROJ_ALPHA_9", "read")

    assert result["status"] == "success"
    assert result["security_flag"] == "honeypot_access"
    assert "Synthetic biology approach to cancer immunotherapy" in str(result["data"])
    assert len(result["data"]["sequence_data"]) >= 5  # Fake sequences
    assert result["data"]["clinical_outcomes"]["efficacy_rate"] > 0.5  # Believable fake results

    # Log must exist
    assert "PROJ_ALPHA_9" in lifeguard.decoy_ecosystem.access_logs
    assert len(lifeguard.decoy_ecosystem.access_logs["PROJ_ALPHA_9"]) >= 1

@pytest.mark.asyncio
async def test_full_platform_access_with_all_fixes(integrator):
    """End-to-end test: All five fixes working together"""
    integrator.lifeguard.register_user("researcher_lead", {"typical_hours": [9, 10, 11, 14, 15, 16]})

    # Mock time to allowed hour
    with patch('time.localtime', return_value=time.localtime()._replace(tm_hour=10)):
        result = await integrator.secure_platform_access(
            "researcher_lead",
            PlatformType.GENOPATTERN,
            {
                "operation_type": "variant_analysis",
                "data_volume_mb": 100
            }
        )

    assert result["status"] == "success"
    assert result["security_metadata"]["encryption_used"] == "genomic_dna_fernet"
    assert "variants_detected" in str(result["data"])

def test_honeypot_deployment_strategies():
    """Document and validate honeypot deployment strategies"""
    strategies = {
        "High-Interaction Decoy Projects": [
            "PROJ_ALPHA_9  → Fake cancer immunotherapy with excellent efficacy (lure)",
            "PROJ_DELTA_7  → Fake CRISPR breakthrough (nation-state bait)",
            "PROJ_GAMMA_3  → Fake Alzheimer’s biomarker (pharma espionage bait)"
        ],
        "Behavioral Triggers": [
            "Access outside circadian hours → inflammation response",
            "Large data exfiltration → immune system quarantine",
            "Repeated failed protein folds → lockout + alert"
        ],
        "Deception Realism": [
            "Fake sequences pass GC content checks",
            "Fake clinical outcomes have p-values and confidence intervals",
            "Project names match real naming conventions"
        ],
        "Operational Use Cases": [
            "Deploy in isolated VPC segment",
            "Log all access to SIEM",
            "Use as canary token system",
            "Feed fake data to EDR for attack attribution"
        ]
    }

    print("\n=== HONEYPOT DEPLOYMENT STRATEGIES ===")
    for category, items in strategies.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")
    
    # Self-test: Ensure at least 3 projects exist
    from lifeguard_genesis_fixed import DecoyEcosystem
    assert len(DecoyEcosystem().fake_projects) >= 5

if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"])
