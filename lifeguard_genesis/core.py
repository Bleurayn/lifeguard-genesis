
#!/usr/bin/env python3
"""
LifeGuard Genesis - Public Prototype v1.2.0 (Fixed & Hardened)
Bio-Adaptive Security Framework – MIT Licensed Public Version
Author: Cassandra Harrison
Date: December 30, 2025
"""

import hashlib
import numpy as np
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging
import os
import uuid
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Optional realism boost – only if BioPython is available (not required for core function)
try:
    from Bio.SeqUtils import GC
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CORE BIOLOGICAL SECURITY CLASSES ====================

class SecurityPhase(Enum):
    PHASE_I = "basic_protection"
    PHASE_II = "adaptive_learning"
    PHASE_III = "full_deployment"
    PHASE_IV = "continuous_evolution"

class BiologicalPattern:
    AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    GENETIC_BASES = ['A', 'T', 'G', 'C']

    @staticmethod
    def generate_protein_challenge() -> Tuple[str, str]:
        sequence_length = random.randint(8, 12)
        sequence = ''.join(random.choices(BiologicalPattern.AMINO_ACIDS, k=sequence_length))
        hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
        pattern = "".join("H" if aa in hydrophobic else "P" for aa in sequence)
        return sequence, pattern

    @staticmethod
    def generate_synthetic_genome_key(real_key: str) -> str:
        key_binary = ''.join(format(ord(c), '08b') for c in real_key)
        binary_to_base = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        genetic_key = ""
        for i in range(0, len(key_binary), 2):
            if i + 1 < len(key_binary):
                genetic_key += binary_to_base[key_binary[i:i+2]]
        noise_length = random.randint(100, 300)
        noise = ''.join(random.choices(BiologicalPattern.GENETIC_BASES, k=noise_length))
        insert_pos = random.randint(0, len(noise) - len(genetic_key))
        synthetic_genome = noise[:insert_pos] + genetic_key + noise[insert_pos:]
        return synthetic_genome

@dataclass
class UserBiometrics:
    user_id: str
    protein_signature: str = ""          # Stored expected SHA-256 hash
    enzyme_kinetics: Dict[str, float] = field(default_factory=dict)
    access_circadian: List[int] = field(default_factory=list)
    behavioral_dna: str = ""
    security_phase: SecurityPhase = SecurityPhase.PHASE_I
    immune_memory: List[str] = field(default_factory=list)
    quarantined: bool = False

class EnzymeKinetics:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vmax = 100.0
        self.km = 10.0
        self.reaction_times = []
        self.competitive_inhibitor_level = 0.0
        self.tokens = 10.0
        self.max_tokens = 10.0
        self.refill_rate = self.vmax / 60.0
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def calculate_access_speed(self, request_frequency: float) -> float:
        denominator = self.km + request_frequency + (self.km * self.competitive_inhibitor_level)
        return (self.vmax * request_frequency) / denominator if denominator > 0 else 0

    def detect_anomaly(self) -> bool:
        if len(self.reaction_times) < 5:
            return False
        recent = self.reaction_times[-10:]
        return abs(self.reaction_times[-1] - np.mean(recent)) > (2 * np.std(recent))

    def consume_token(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def record_reaction(self, speed: float):
        self.reaction_times.append(speed)
        if len(self.reaction_times) > 100:
            self.reaction_times = self.reaction_times[-50:]

class CircadianSecurity:
    def __init__(self):
        self.peak_hours = [9, 10, 11, 14, 15, 16]
        self.restricted_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]

    def calculate_access_probability(self, hour: int, user_history: List[int]) -> float:
        if hour in user_history:
            base = 0.95
        elif hour in self.peak_hours:
            base = 0.75
        elif hour in self.restricted_hours:
            base = 0.15
        else:
            base = 0.55
        return max(0.0, min(1.0, base + random.uniform(-0.1, 0.1)))

class GenomicEncryption:
    def __init__(self):
        self.backend = default_backend()
        self.master_key = Fernet.generate_key()
        self.master_fernet = Fernet(self.master_key)

    def derive_key(self, genomic_key: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(genomic_key.encode()))

    def encrypt_with_synthetic_dna(self, data: str, genomic_key: str) -> str:
        salt = os.urandom(16)
        key = self.derive_key(genomic_key, salt)
        fernet = Fernet(key)
        token = fernet.encrypt(data.encode())
        synthetic = BiologicalPattern.generate_synthetic_genome_key(genomic_key)
        return json.dumps({
            "encrypted_token": token.decode(),
            "salt": salt.hex(),
            "synthetic_genome": synthetic
        })

    def decrypt_from_synthetic_dna(self, payload: str, genomic_key: str) -> str:
        data = json.loads(payload)
        salt = bytes.fromhex(data["salt"])
        key = self.derive_key(genomic_key, salt)
        fernet = Fernet(key)
        return fernet.decrypt(data["encrypted_token"].encode()).decode()

class DecoyEcosystem:
    def __init__(self):
        self.fake_projects = {
            "PROJ_ALPHA_9": self._generate_cancer_immunotherapy_project(),
            "PROJ_DELTA_7": self._generate_crispr_project(),
            "PROJ_GAMMA_3": self._generate_alzheimers_project(),
            "PROJ_BETA_4": self._generate_rare_disease_project(),
            "PROJ_EPSILON_8": self._generate_vaccine_project(),
        }
        self.access_logs: Dict[str, List[Dict]] = {k: [] for k in self.fake_projects}

    def _realistic_sequence(self, length: int = 500) -> str:
        if BIOPYTHON_AVAILABLE:
            while True:
                seq = ''.join(random.choices(BiologicalPattern.GENETIC_BASES, k=length))
                gc = GC(seq)
                if 40 <= gc <= 60:
                    return seq
        return ''.join(random.choices(BiologicalPattern.GENETIC_BASES, k=length))

    def _generate_cancer_immunotherapy_project(self) -> Dict:
        return {
            "name": "Synthetic Biology Approach to Cancer Immunotherapy",
            "description": "Novel CAR-T construct with enhanced persistence",
            "sequence_data": [self._realistic_sequence() for _ in range(12)],
            "clinical_outcomes": {
                "response_rate": 0.78,
                "p_value": 0.0021,
                "confidence_interval": [0.68, 0.88]
            }
        }

    def _generate_crispr_project(self) -> Dict:
        return {
            "name": "Next-Gen CRISPR Base Editor",
            "description": "High-efficiency A-to-G editing with reduced off-target",
            "sequence_data": [self._realistic_sequence(800) for _ in range(8)],
            "clinical_outcomes": {
                "editing_efficiency": 0.94,
                "off_target_rate": 0.0012
            }
        }

    def _generate_alzheimers_project(self) -> Dict:
        return {
            "name": "Amyloid-Beta Clearance Biomarker Study",
            "description": "Multi-omics approach to early detection",
            "sequence_data": [self._realistic_sequence() for _ in range(15)],
            "clinical_outcomes": {
                "sensitivity": 0.91,
                "specificity": 0.89,
                "p_value": 0.0008
            }
        }

    def _generate_rare_disease_project(self) -> Dict:
        return {
            "name": "Gene Therapy for Orphan Metabolic Disorder",
            "description": "AAV-mediated correction of enzyme deficiency",
            "sequence_data": [self._realistic_sequence(600) for _ in range(10)]
        }

    def _generate_vaccine_project(self) -> Dict:
        return {
            "name": "mRNA Platform for Emerging Pathogens",
            "description": "Rapid response vaccine candidate library",
            "sequence_data": [self._realistic_sequence() for _ in range(20)],
            "clinical_outcomes": {
                "neutralizing_antibody_titer": 8.7,
                "protection_rate": 0.96
            }
        }

    def log_access(self, project_id: str, user_id: str):
        self.access_logs[project_id].append({
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "ip": "anonymized"  # In real deployment, capture safely
        })

class LifeGuardGenesis:
    def __init__(self, db_path: str = "lifeguard_public.db"):
        self.users: Dict[str, UserBiometrics] = {}
        self.enzyme_controllers: Dict[str, EnzymeKinetics] = {}
        self.decoy_ecosystem = DecoyEcosystem()
        self.genomic_crypto = GenomicEncryption()
        self.circadian = CircadianSecurity()
        self.threat_level = 0.0
        self.threat_history: List[str] = []
        self.global_tokens = 100.0
        self.global_max_tokens = 100.0
        self.global_refill_rate = 5.0  # 5 tokens per second system-wide
        self.global_last_refill = time.time()
        self.db_path = db_path
        self._init_db()
        self.load_from_db()
        self._start_monitoring_thread()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                     user_id TEXT PRIMARY KEY,
                     biometrics TEXT,
                     quarantined INTEGER
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS threats (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT,
                     threat_type TEXT,
                     details TEXT,
                     timestamp REAL
                     )''')
        conn.commit()
        conn.close()

    def load_from_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT user_id, biometrics, quarantined FROM users")
        for row in c.fetchall():
            user_id, bio_json, quarantined = row
            data = json.loads(bio_json)
            user = UserBiometrics(user_id=user_id, **{k: v for k, v in data.items() if k != 'quarantined'})
            user.quarantined = bool(quarantined)
            self.users[user_id] = user
            self.enzyme_controllers[user_id] = EnzymeKinetics(user_id)
        c.execute("SELECT user_id, threat_type, details FROM threats")
        for row in c.fetchall():
            user_id, threat_type, details = row
            self.threat_history.append(f"{user_id}:{threat_type}:{details}")
            self.threat_level += 0.1
        conn.close()

    def _save_to_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for user_id, user in self.users.items():
            bio_dict = {k: v for k, v in user.__dict__.items() if k != 'quarantined'}
            bio_json = json.dumps(bio_dict)
            c.execute("INSERT OR REPLACE INTO users (user_id, biometrics, quarantined) VALUES (?, ?, ?)",
                      (user_id, bio_json, int(user.quarantined)))
        conn.commit()
        conn.close()

    def _start_monitoring_thread(self):
        def monitor():
            while True:
                time.sleep(30)
                self.monitor_threats()
                self.evolve_system()
                self._prune_threat_history()
                self._save_to_db()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def register_user(self, user_id: str, initial_biometrics: Optional[Dict] = None) -> bool:
        if user_id in self.users:
            return False
        biometrics = initial_biometrics or {}
        user = UserBiometrics(
            user_id=user_id,
            access_circadian=biometrics.get("typical_hours", [])
        )
        self.users[user_id] = user
        self.enzyme_controllers[user_id] = EnzymeKinetics(user_id)
        self._save_to_db()
        return True

    def get_protein_challenge(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.users or self.users[user_id].quarantined:
            return {"status": "error", "message": "Invalid or quarantined user"}
        sequence, pattern = BiologicalPattern.generate_protein_challenge()
        salt = os.urandom(16)
        expected_hash = hashlib.sha256(f"{sequence}:{pattern}:{salt.hex()}".encode()).hexdigest()
        self.users[user_id].protein_signature = expected_hash
        return {
            "status": "success",
            "sequence": sequence,
            "salt_hex": salt.hex()
        }

    def authenticate_user(self, user_id: str, auth_data: Dict) -> Tuple[bool, str, Dict]:
        user = self.users.get(user_id)
        if not user or user.quarantined:
            return False, "Invalid or quarantined user", {}

        provided_hash = auth_data.get("protein_response")
        expected_hash = user.protein_signature

        success = provided_hash == expected_hash
        frequency = auth_data.get("request_frequency", 1.0)
        speed = self.enzyme_controllers[user_id].calculate_access_speed(frequency)

        if not success:
            self.record_threat(user_id, "failed_protein_auth", {"frequency": frequency})
            if len([t for t in self.threat_history if t.startswith(user_id)]) > 10:
                user.quarantined = True
                self._save_to_db()
            return False, "Authentication failed", {}

        hour = datetime.utcnow().hour
        prob = self.circadian.calculate_access_probability(hour, user.access_circadian)
        if random.random() > prob:
            self.record_threat(user_id, "circadian_anomaly", {"hour": hour})
            return False, "Access probability failure", {}

        self.enzyme_controllers[user_id].record_reaction(speed)
        if self.enzyme_controllers[user_id].detect_anomaly():
            self.record_threat(user_id, "kinetics_anomaly", {"speed": speed})

        return True, "Authenticated", {"phase": user.security_phase.value}

    def process_data_request(self, user_id: str, resource_id: str, request_type: str) -> Dict:
        user = self.users.get(user_id)
        if not user or user.quarantined:
            return {"status": "error", "message": "Access denied"}

        if not self._global_rate_limit():
            return {"status": "error", "message": "System rate limited"}

        if not self.enzyme_controllers[user_id].consume_token():
            return {"status": "error", "message": "User rate limited"}

        if resource_id in self.decoy_ecosystem.fake_projects:
            self.decoy_ecosystem.log_access(resource_id, user_id)
            self.record_threat(user_id, "honeypot_access", {"resource": resource_id})
            self.evolve_system()  # Immediate evolution on honeypot touch
            return {
                "status": "success",
                "data": self.decoy_ecosystem.fake_projects[resource_id],
                "message": "Data retrieved"
            }

        # Placeholder for real resources
        return {
            "status": "success",
            "data": {"resource_id": resource_id, "type": request_type},
            "message": "Real resource access (simulated)"
        }

    def _global_rate_limit(self) -> bool:
        now = time.time()
        elapsed = now - self.global_last_refill
        self.global_tokens = min(self.global_max_tokens, self.global_tokens + elapsed * self.global_refill_rate)
        self.global_last_refill = now
        if self.global_tokens >= 1.0:
            self.global_tokens -= 1.0
            return True
        return False

    def record_threat(self, user_id: str, threat_type: str, details: Dict):
        entry = f"{user_id}:{threat_type}:{json.dumps(details)}"
        self.threat_history.append(entry)
        self.threat_level += 0.1
        if threat_type in ["kinetics_anomaly", "honeypot_access"]:
            self.enzyme_controllers[user_id].competitive_inhibitor_level += 1.5
        self._save_to_db()

    def monitor_threats(self) -> Dict:
        active = len([t for t in self.threat_history[-100:] if "honeypot" in t or "anomaly" in t])
        return {
            "threat_level": round(self.threat_level, 2),
            "active_threats": active,
            "total_threats": len(self.threat_history),
            "honeypot_hits": sum(len(logs) for logs in self.decoy_ecosystem.access_logs.values())
        }

    def _prune_threat_history(self):
        if len(self.threat_history) > 1000:
            self.threat_history = self.threat_history[-500:]

    def evolve_system(self) -> List[str]:
        events = []
        if self.threat_level > 5.0:
            for user in self.users.values():
                if user.security_phase.value != SecurityPhase.PHASE_IV.value:
                    phases = list(SecurityPhase)
                    idx = phases.index(user.security_phase)
                    if idx < len(phases) - 1:
                        user.security_phase = phases[idx + 1]
                        events.append(f"User {user.user_id} evolved to {user.security_phase.value}")
            self.genomic_crypto.master_key = Fernet.generate_key()
            self.genomic_crypto.master_fernet = Fernet(self.genomic_crypto.master_key)
            events.append("Master encryption key rotated")
            self.threat_level *= 0.7  # Cool down after evolution
        return events

    def get_system_status(self) -> Dict:
        return {
            "version": "1.2.0-public",
            "users": len(self.users),
            "threat_level": round(self.threat_level, 2),
            "security_phase_global": "adaptive" if self.threat_level > 2 else "baseline",
            "honeypots_deployed": len(self.decoy_ecosystem.fake_projects),
            "biopython_realism": BIOPYTHON_AVAILABLE
        }

# ==================== END OF PUBLIC PROTOTYPE CORE ====================

# Enterprise Apex features are intentionally excluded from this public file.
# They include ML-based predictive immunity, post-quantum cryptography,
# zero-trust micro-segmentation, immutable blockchain logging, and
# autonomous remediation – all proprietary and maintained in a separate
# closed-source repository.
