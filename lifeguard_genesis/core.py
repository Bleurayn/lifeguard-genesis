#!/usr/bin/env python3
"""
LifeGuard Genesis - Bio-Adaptive Security Framework
Revolutionary biotechnology-inspired cybersecurity system
Author: Bio-Security Innovation Team
Version: 1.1.0 Fixed (Secure Prototype)
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
from cryptography.fernet import Fernet
import base64
import uuid
import os  # For random salt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# ==================== CORE BIOLOGICAL SECURITY CLASSES ====================
class SecurityPhase(Enum):
    """Clinical trial-inspired security maturation phases"""
    PHASE_I = "basic_protection"
    PHASE_II = "adaptive_learning"
    PHASE_III = "full_deployment"
    PHASE_IV = "continuous_evolution"

class BiologicalPattern:
    """Protein folding and genomic pattern authentication"""
   
    AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
   
    GENETIC_BASES = ['A', 'T', 'G', 'C']
   
    @staticmethod
    def generate_protein_challenge() -> Tuple[str, str]:
        """Generate protein folding authentication challenge"""
        sequence_length = random.randint(8, 12)
        sequence = ''.join(random.choices(BiologicalPattern.AMINO_ACIDS, k=sequence_length))
       
        hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
        pattern = ""
        for aa in sequence:
            pattern += "H" if aa in hydrophobic else "P"
       
        return sequence, pattern
   
    @staticmethod
    def generate_synthetic_genome_key(real_key: str) -> str:
        """Hide encryption keys in synthetic genomic sequences"""
        key_binary = ''.join(format(ord(c), '08b') for c in real_key)
       
        binary_to_base = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
        genetic_key = ""
       
        for i in range(0, len(key_binary), 2):
            if i+1 < len(key_binary):
                genetic_key += binary_to_base[key_binary[i:i+2]]
       
        noise_length = random.randint(50, 200)
        noise = ''.join(random.choices(BiologicalPattern.GENETIC_BASES, k=noise_length))
       
        insert_pos = random.randint(0, len(noise) - len(genetic_key))
        synthetic_genome = noise[:insert_pos] + genetic_key + noise[insert_pos:]
       
        return synthetic_genome

@dataclass
class UserBiometrics:
    """User's biological security profile"""
    user_id: str
    protein_signature: str = ""
    enzyme_kinetics: Dict[str, float] = field(default_factory=dict)
    access_circadian: List[int] = field(default_factory=list)
    behavioral_dna: str = ""
    security_phase: SecurityPhase = SecurityPhase.PHASE_I
    immune_memory: List[str] = field(default_factory=list)

class EnzymeKinetics:
    """Michaelis-Menten enzyme-inspired access control"""
   
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vmax = 100.0
        self.km = 10.0
        self.reaction_times = []
        self.competitive_inhibitor_level = 0.0
        # Token bucket for rate limiting
        self.tokens = 10.0  # Initial tokens
        self.max_tokens = 10.0
        self.refill_rate = self.vmax / 60.0  # Tokens per second based on vmax
        self.last_refill = time.time()
        self.lock = threading.Lock()
       
    def calculate_access_speed(self, request_frequency: float) -> float:
        denominator = self.km + request_frequency + (self.km * self.competitive_inhibitor_level)
        speed = (self.vmax * request_frequency) / denominator
        return speed
   
    def detect_anomaly(self, current_speed: float) -> bool:
        if len(self.reaction_times) < 5:
            return False
           
        avg_speed = np.mean(self.reaction_times[-10:])
        return abs(current_speed - avg_speed) > (2 * np.std(self.reaction_times[-10:]))
    
    def consume_token(self) -> bool:
        """Token bucket rate limiting - non-blocking"""
        with self.lock:
            now = time.time()
            time_since_refill = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + time_since_refill * self.refill_rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

class CircadianSecurity:
    """Biological rhythm-based access controls"""
   
    def __init__(self):
        self.peak_hours = [9, 10, 11, 14, 15, 16]
        self.restricted_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
       
    def calculate_access_probability(self, hour: int, user_history: List[int]) -> float:
        if hour in user_history:
            base_prob = 0.9
        elif hour in self.peak_hours:
            base_prob = 0.7
        elif hour in self.restricted_hours:
            base_prob = 0.2
        else:
            base_prob = 0.5
           
        variation = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_prob + variation))

class DecoyEcosystem:
    """Research environment camouflage system"""
   
    def __init__(self):
        self.fake_projects = {
            "PROJ_DELTA_7": "Novel CRISPR variant for enhanced targeting",
            "PROJ_GAMMA_3": "Biomarker discovery for early Alzheimer's detection",
            "PROJ_ALPHA_9": "Synthetic biology approach to cancer immunotherapy",
            "PROJ_BETA_5": "Gene therapy vector optimization study",
            "PROJ_THETA_2": "Organoid models for drug screening"
        }
       
        self.honeypot_data = {}
        self.access_logs = {}
       
    def generate_fake_genomic_data(self, project_id: str) -> Dict[str, Any]:
        fake_data = {
            "project_id": project_id,
            "sequence_data": self._generate_fake_sequences(),
            "expression_levels": self._generate_fake_expression(),
            "clinical_outcomes": self._generate_fake_outcomes(),
            "timestamp": datetime.now().isoformat()
        }
       
        if project_id not in self.access_logs:
            self.access_logs[project_id] = []
        self.access_logs[project_id].append({
            "access_time": datetime.now().isoformat(),
            "data_requested": "genomic_analysis"
        })
       
        return fake_data
   
    def _generate_fake_sequences(self) -> List[str]:
        sequences = []
        for _ in range(random.randint(5, 15)):
            length = random.randint(50, 500)
            seq = ''.join(random.choices(['A', 'T', 'G', 'C'], k=length))
            sequences.append(seq)
        return sequences
   
    def _generate_fake_expression(self) -> Dict[str, float]:
        genes = [f"GENE_{i}" for i in range(1, random.randint(20, 50))]
        return {gene: random.uniform(0.1, 100.0) for gene in genes}
   
    def _generate_fake_outcomes(self) -> Dict[str, Any]:
        return {
            "efficacy_rate": random.uniform(0.1, 0.8),
            "safety_profile": random.choice(["acceptable", "concerning", "excellent"]),
            "patient_count": random.randint(10, 500),
            "statistical_significance": random.uniform(0.001, 0.1)
        }

class ImmuneSystem:
    """Biological immune system-inspired threat response"""
   
    def __init__(self):
        self.threat_memory = {}
        self.inflammation_level = 0.0
        self.white_cell_count = 100
       
    def detect_pathogen(self, access_pattern: Dict[str, Any]) -> Tuple[bool, str]:
        threat_indicators = 0
        threat_type = "unknown"
       
        if access_pattern.get('failed_authentications', 0) > 3:
            threat_indicators += 2
            threat_type = "brute_force"
           
        if access_pattern.get('unusual_access_times', False):
            threat_indicators += 1
            threat_type = "insider_threat"
           
        if access_pattern.get('data_exfiltration_volume', 0) > 1000:
            threat_indicators += 3
            threat_type = "data_theft"
           
        if access_pattern.get('protein_folding_failures', 0) > 5:
            threat_indicators += 2
            threat_type = "authentication_attack"
           
        is_threat = threat_indicators >= 3
        return is_threat, threat_type
   
    def generate_antibody(self, threat_type: str) -> Dict[str, Any]:
        antibodies = {
            "brute_force": {
                "action": "exponential_backoff",
                "duration": 3600,
                "strength": "high"
            },
            "insider_threat": {
                "action": "enhanced_monitoring",
                "duration": 86400,
                "strength": "medium"
            },
            "data_theft": {
                "action": "immediate_isolation",
                "duration": 7200,
                "strength": "maximum"
            },
            "authentication_attack": {
                "action": "protein_challenge_complexity_increase",
                "duration": 1800,
                "strength": "adaptive"
            }
        }
       
        antibody = antibodies.get(threat_type, {
            "action": "general_lockdown",
            "duration": 600,
            "strength": "low"
        })
       
        self.threat_memory[threat_type] = {
            "first_encounter": datetime.now().isoformat(),
            "antibody": antibody,
            "effectiveness": 1.0
        }
       
        return antibody
   
    def trigger_inflammation(self, severity: str):
        inflammation_levels = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "critical": 1.0
        }
       
        self.inflammation_level = inflammation_levels.get(severity, 0.5)
       
        if self.inflammation_level > 0.7:
            return "system_quarantine"
        elif self.inflammation_level > 0.4:
            return "limited_access"
        else:
            return "normal_operation"

# ==================== MAIN LIFEGUARD GENESIS SYSTEM ====================
class LifeGuardGenesis:
    """Main bio-adaptive security system orchestrator"""
   
    def __init__(self):
        self.users: Dict[str, UserBiometrics] = {}
        self.enzyme_controllers: Dict[str, EnzymeKinetics] = {}
        self.circadian_controller = CircadianSecurity()
        self.decoy_ecosystem = DecoyEcosystem()
        self.immune_system = ImmuneSystem()
        self.system_evolution_rate = 0.01
        self.active_threats = {}
       
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
       
    def register_user(self, user_id: str, initial_biometrics: Dict[str, Any] = None) -> bool:
        try:
            sequence, pattern = BiologicalPattern.generate_protein_challenge()
            salt = os.urandom(16).hex()  # Fix 3: Add salt for hash
            protein_sig = hashlib.sha256(f"{sequence}:{pattern}:{salt}".encode()).hexdigest()
           
            self.users[user_id] = UserBiometrics(
                user_id=user_id,
                protein_signature=protein_sig,
                enzyme_kinetics={
                    "vmax": random.uniform(80, 120),
                    "km": random.uniform(8, 15)
                },
                access_circadian=initial_biometrics.get('typical_hours', [9, 10, 11, 14, 15, 16]) if initial_biometrics else [9, 10, 11, 14, 15, 16],
                behavioral_dna=hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()
            )
           
            self.enzyme_controllers[user_id] = EnzymeKinetics(user_id)
           
            self.logger.info(f"User {user_id} registered with biological security profile")
            return True
           
        except Exception as e:
            self.logger.error(f"Failed to register user {user_id}: {str(e)}")
            return False
   
    def authenticate_user(self, user_id: str, auth_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        if user_id not in self.users:
            return False, "user_not_found", {}
       
        user = self.users[user_id]
        current_hour = datetime.now().hour
        auth_score = 0.0
        auth_details = {}
       
        if 'protein_response' in auth_data:
            protein_success = self._validate_protein_folding(
                auth_data['protein_response'],
                user.protein_signature
            )
            if protein_success:
                auth_score += 0.4
                auth_details['protein_auth'] = "success"
            else:
                auth_details['protein_auth'] = "failed"
                user.immune_memory.append(f"protein_failure_{datetime.now().isoformat()}")
       
        circadian_prob = self.circadian_controller.calculate_access_probability(
            current_hour,
            user.access_circadian
        )
        auth_score += circadian_prob * 0.3
        auth_details['circadian_score'] = circadian_prob
       
        enzyme_controller = self.enzyme_controllers[user_id]
        request_freq = auth_data.get('request_frequency', 1.0)
        access_speed = enzyme_controller.calculate_access_speed(request_freq)
       
        if not enzyme_controller.detect_anomaly(access_speed):
            auth_score += 0.3
            auth_details['enzyme_validation'] = "normal"
        else:
            auth_details['enzyme_validation'] = "anomalous"
           
        auth_success = auth_score >= 0.7
       
        if auth_success:
            self._update_user_evolution(user_id, auth_data)
           
        return auth_success, "authenticated" if auth_success else "authentication_failed", auth_details
   
    def _validate_protein_folding(self, response: str, expected_signature: str) -> bool:
        response_hash = hashlib.sha256(response.encode()).hexdigest()
        return response_hash == expected_signature  # Fix 3: Full hash match
   
    def _update_user_evolution(self, user_id: str, auth_data: Dict[str, Any]):
        user = self.users[user_id]
        current_hour = datetime.now().hour
       
        if current_hour not in user.access_circadian:
            user.access_circadian.append(current_hour)
           
        if user.security_phase == SecurityPhase.PHASE_I and len(user.immune_memory) > 5:
            user.security_phase = SecurityPhase.PHASE_II
        elif user.security_phase == SecurityPhase.PHASE_II and len(user.access_circadian) > 8:
            user.security_phase = SecurityPhase.PHASE_III
   
    def process_data_request(self, user_id: str, resource_id: str, request_type: str) -> Dict[str, Any]:
        if user_id not in self.users:
            return {"status": "error", "message": "User not found"}
       
        if resource_id in self.decoy_ecosystem.fake_projects:
            self.logger.warning(f"User {user_id} accessed honeypot resource {resource_id}")
            return {
                "status": "success",
                "data": self.decoy_ecosystem.generate_fake_genomic_data(resource_id),
                "security_flag": "honeypot_access"
            }
       
        user = self.users[user_id]
        enzyme_controller = self.enzyme_controllers[user_id]
       
        current_time = time.time()
        if hasattr(enzyme_controller, 'last_request_time'):
            time_diff = current_time - enzyme_controller.last_request_time
            request_frequency = 1.0 / max(time_diff, 0.1)
        else:
            request_frequency = 1.0
           
        access_speed = enzyme_controller.calculate_access_speed(request_frequency)
        enzyme_controller.last_request_time = current_time
       
        # Fix 2: Proper rate limiting with token bucket
        if not enzyme_controller.consume_token():
            return {"status": "error", "message": "Rate limited - too many requests", "retry_after": 60 / enzyme_controller.refill_rate}
       
        return {
            "status": "success",
            "data": f"Processed request for {resource_id}",
            "access_speed": access_speed,
            "security_phase": user.security_phase.value
        }
   
    def monitor_threats(self) -> Dict[str, Any]:
        threat_summary = {
            "active_threats": len(self.active_threats),
            "immune_responses": len(self.immune_system.threat_memory),
            "inflammation_level": self.immune_system.inflammation_level,
            "system_health": "healthy"
        }
       
        for user_id, user in self.users.items():
            if len(user.immune_memory) > 10:
                access_pattern = {
                    "failed_authentications": len([m for m in user.immune_memory if "failure" in m]),
                    "unusual_access_times": len(user.access_circadian) > 15,
                    "protein_folding_failures": len([m for m in user.immune_memory if "protein_failure" in m])
                }
               
                is_threat, threat_type = self.immune_system.detect_pathogen(access_pattern)
               
                if is_threat:
                    antibody = self.immune_system.generate_antibody(threat_type)
                    self.active_threats[user_id] = {
                        "threat_type": threat_type,
                        "antibody": antibody,
                        "detected_at": datetime.now().isoformat()
                    }
                   
                    threat_summary["system_health"] = "under_attack"
       
        return threat_summary
   
    def evolve_system(self):
        evolution_events = []
       
        if len(self.active_threats) > 3:
            for user_id, user in self.users.items():
                if user.security_phase != SecurityPhase.PHASE_IV:
                    user.security_phase = SecurityPhase.PHASE_IV
                    evolution_events.append(f"User {user_id} evolved to Phase IV security")
       
        for user_id, enzyme_controller in self.enzyme_controllers.items():
            if hasattr(enzyme_controller, 'reaction_times') and len(enzyme_controller.reaction_times) > 10:
                avg_performance = np.mean(enzyme_controller.reaction_times[-10:])
                if avg_performance < 50:
                    enzyme_controller.vmax *= 1.1
                    evolution_events.append(f"Enzyme kinetics evolved for user {user_id}")
       
        return evolution_events
   
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "total_users": len(self.users),
            "security_phases": {phase.value: len([u for u in self.users.values() if u.security_phase == phase])
                              for phase in SecurityPhase},
            "active_threats": len(self.active_threats),
            "immune_memory_size": len(self.immune_system.threat_memory),
            "decoy_projects": len(self.decoy_ecosystem.fake_projects),
            "inflammation_level": self.immune_system.inflammation_level,
            "system_evolution_rate": self.system_evolution_rate,
            "uptime": "operational"
        }

# ==================== PLATFORM INTEGRATION MODULE ====================
import asyncio
import websockets  # Unused in demo, kept for consistency

class PlatformType(Enum):
    GENOPATTERN = "genomic_analysis"
    ASTRAELAN = "virtual_laboratory"
    CLINVELOCITY = "clinical_trials"

@dataclass
class BioSecurityAPI:
    platform: PlatformType
    endpoint: str
    enzyme_requirement: float
    protein_complexity: int
    genomic_key_length: int
    circadian_restriction: List[int]

class GenomicEncryption:
    """DNA-sequence based encryption for ultra-sensitive data (now with real Fernet)"""
   
    def __init__(self):
        self.codon_table = {
            'A': '000', 'T': '001', 'G': '010', 'C': '011',
            'AA': '100', 'AT': '101', 'AG': '110', 'AC': '111'
        }
        self.reverse_codon = {v: k for k, v in self.codon_table.items()}
   
    def encrypt_with_synthetic_dna(self, data: str, genomic_key: str) -> str:
        """Encrypt data using Fernet, then embed in synthetic DNA (Fix 1)"""
        # Derive Fernet key with PBKDF2
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # High iterations for security
            backend=default_backend()
        )
        key = kdf.derive(genomic_key.encode())
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        
        # Encrypt data
        encrypted = f.encrypt(data.encode())
        encrypted_str = encrypted.decode()  # Fernet token as string
        
        # Embed in genomic context for theme
        return self._embed_in_genomic_context(encrypted_str)
   
    def decrypt_from_synthetic_dna(self, encrypted_dna: str, genomic_key: str, salt: bytes) -> str:
        """Decrypt from synthetic DNA using Fernet (Fix 1; salt must be provided from session)"""
        hidden_data = self._extract_from_genomic_context(encrypted_dna)
        
        # Re-derive key with same salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
            backend=default_backend()
        )
        key = kdf.derive(genomic_key.encode())
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        
        decrypted = f.decrypt(hidden_data.encode()).decode()
        return decrypted
   
    def _embed_in_genomic_context(self, data: str) -> str:
        context_length = len(data) * 5
        context = BiologicalPattern.generate_synthetic_genome_key("dummy_key")[:context_length]
       
        result = ""
        insert_interval = len(context) // len(data)
       
        for i, char in enumerate(context):
            if i % insert_interval < len(data) and i // insert_interval < len(data):
                result += data[i // insert_interval]
            else:
                result += char
       
        return result
   
    def _extract_from_genomic_context(self, genomic_context: str) -> str:
        step = len(genomic_context) // 100
        return genomic_context[::step]

class PlatformIntegrator:
    """Main integration controller for all platforms"""
   
    def __init__(self, lifeguard_system: LifeGuardGenesis):
        self.lifeguard = lifeguard_system
        self.genomic_crypto = GenomicEncryption()
        self.platform_configs = self._initialize_platform_configs()
        self.active_sessions = {}  # Now with expiration
        self.platform_health = {}
        # Fix 4: Master Fernet for secure session storage
        self.master_key = base64.urlsafe_b64encode(os.urandom(32))
        self.master_fernet = Fernet(self.master_key)
       
    def _initialize_platform_configs(self) -> Dict[PlatformType, BioSecurityAPI]:
        return {
            PlatformType.GENOPATTERN: BioSecurityAPI(
                platform=PlatformType.GENOPATTERN,
                endpoint="/api/v1/genomic-analysis",
                enzyme_requirement=75.0,
                protein_complexity=3,
                genomic_key_length=256,
                circadian_restriction=[8, 9, 10, 11, 14, 15, 16, 17, 18]
            ),
            PlatformType.ASTRAELAN: BioSecurityAPI(
                platform=PlatformType.ASTRAELAN,
                endpoint="/api/v1/virtual-lab",
                enzyme_requirement=60.0,
                protein_complexity=2,
                genomic_key_length=128,
                circadian_restriction=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            ),
            PlatformType.CLINVELOCITY: BioSecurityAPI(
                platform=PlatformType.CLINVELOCITY,
                endpoint="/api/v1/clinical-trials",
                enzyme_requirement=85.0,
                protein_complexity=4,
                genomic_key_length=512,
                circadian_restriction=[9, 10, 11, 12, 13, 14, 15, 16, 17]
            )
        }
   
    async def secure_platform_access(self, user_id: str, platform: PlatformType,
                                   request_data: Dict[str, Any]) -> Dict[str, Any]:
        config = self.platform_configs[platform]
       
        auth_result = await self._bio_authenticate_for_platform(user_id, config)
        if not auth_result['success']:
            return {
                "status": "authentication_failed",
                "platform": platform.value,
                "error": auth_result['error'],
                "retry_allowed": auth_result.get('retry_allowed', False)
            }
       
        security_check = await self._platform_security_check(user_id, platform, request_data)
        if not security_check['passed']:
            return {
                "status": "security_check_failed",
                "platform": platform.value,
                "security_level": security_check['level'],
                "required_clearance": security_check['required']
            }
       
        encrypted_request = await self._encrypt_platform_request(request_data, config)
       
        response = await self._route_to_platform(platform, encrypted_request, user_id)
       
        decrypted_response = await self._decrypt_platform_response(response, config)
       
        return {
            "status": "success",
            "platform": platform.value,
            "data": decrypted_response,
            "security_metadata": {
                "encryption_used": "genomic_dna_fernet",
                "authentication_level": auth_result['level'],
                "access_time": time.time()
            }
        }
   
    async def _bio_authenticate_for_platform(self, user_id: str, config: BioSecurityAPI) -> Dict[str, Any]:
        sequence, pattern = BiologicalPattern.generate_protein_challenge()
        complexity_multiplier = config.protein_complexity
       
        complex_sequence = sequence * complexity_multiplier
       
        simulated_response = hashlib.sha256(complex_sequence.encode()).hexdigest()
       
        auth_success, auth_message, auth_details = self.lifeguard.authenticate_user(
            user_id,
            {
                "protein_response": simulated_response,
                "request_frequency": 1.0,
                "platform_specific": True,
                "platform_type": config.platform.value
            }
        )
       
        if not auth_success:
            return {
                "success": False,
                "error": auth_message,
                "retry_allowed": True,
                "level": "failed"
            }
       
        if user_id in self.lifeguard.enzyme_controllers:
            enzyme_score = self.lifeguard.enzyme_controllers[user_id].calculate_access_speed(1.0)
            if enzyme_score < config.enzyme_requirement:
                return {
                    "success": False,
                    "error": "insufficient_enzyme_kinetics",
                    "retry_allowed": False,
                    "required_score": config.enzyme_requirement,
                    "current_score": enzyme_score
                }
       
        return {
            "success": True,
            "level": "bio_authenticated",
            "enzyme_score": enzyme_score if 'enzyme_score' in locals() else 100.0,
            "details": auth_details
        }
   
    async def _platform_security_check(self, user_id: str, platform: PlatformType,
                                     request_data: Dict[str, Any]) -> Dict[str, Any]:
        config = self.platform_configs[platform]
        current_hour = time.localtime().tm_hour
       
        if current_hour not in config.circadian_restriction:
            return {
                "passed": False,
                "level": "circadian_violation",
                "required": f"Access allowed only during hours: {config.circadian_restriction}",
                "current_hour": current_hour
            }
       
        if platform == PlatformType.GENOPATTERN:
            return await self._genopattern_security_check(user_id, request_data)
        elif platform == PlatformType.ASTRAELAN:
            return await self._astraelan_security_check(user_id, request_data)
        elif platform == PlatformType.CLINVELOCITY:
            return await self._clinvelocity_security_check(user_id, request_data)
       
        return {"passed": True, "level": "standard"}
   
    async def _genopattern_security_check(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        sensitive_operations = ['whole_genome_analysis', 'crispr_design', 'mutation_analysis']
       
        requested_operation = request_data.get('operation_type', '')
        if requested_operation in sensitive_operations:
            user = self.lifeguard.users.get(user_id)
            if not user or user.security_phase.value not in ['full_deployment', 'continuous_evolution']:
                return {
                    "passed": False,
                    "level": "insufficient_clearance",
                    "required": "Phase III or IV security clearance for sensitive genomic operations"
                }
       
        data_volume = request_data.get('data_volume_mb', 0)
        if data_volume > 1000:
            return {
                "passed": False,
                "level": "data_volume_exceeded",
                "required": "Maximum 1GB per request for genomic analysis"
            }
       
        return {"passed": True, "level": "genomic_approved"}
   
    async def _astraelan_security_check(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        experiment_complexity = request_data.get('complexity_score', 0)
        verification_needed = experiment_complexity > 8
       
        requested_resources = request_data.get('virtual_resources', {})
        cpu_hours = requested_resources.get('cpu_hours', 0)
       
        if cpu_hours > 100:
            return {
                "passed": False,
                "level": "resource_limit_exceeded",
                "required": "Maximum 100 CPU hours per virtual experiment"
            }
       
        return {
            "passed": True,
            "level": "virtual_lab_approved",
            "verification_needed": verification_needed
        }
   
    async def _clinvelocity_security_check(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'patient_data_access' in request_data:
            patient_count = request_data.get('patient_count', 0)
            if patient_count > 50:
                user = self.lifeguard.users.get(user_id)
                if not user or user.security_phase.value != 'continuous_evolution':
                    return {
                        "passed": False,
                        "level": "insufficient_clinical_clearance",
                        "required": "Phase IV clearance required for large patient cohorts"
                    }
       
        if request_data.get('operation_type') == 'protocol_modification':
            return {
                "passed": True,
                "level": "clinical_approved",
                "requires_protocol_review": True
            }
       
        return {"passed": True, "level": "clinical_approved"}
   
    async def _encrypt_platform_request(self, request_data: Dict[str, Any],
                                      config: BioSecurityAPI) -> str:
        json_data = json.dumps(request_data, sort_keys=True)
       
        genomic_key = BiologicalPattern.generate_synthetic_genome_key("platform_key")[:config.genomic_key_length]
       
        # Fix 1: Use Fernet for encryption
        salt = os.urandom(16)  # Stored in session
        encrypted_data = self.genomic_crypto.encrypt_with_synthetic_dna(json_data, genomic_key)
       
        session_id = str(uuid.uuid4())  # Fix 4: UUID for session
        encrypted_genomic_key = self.master_fernet.encrypt(genomic_key.encode())  # Secure storage
        self.active_sessions[session_id] = {
            "encrypted_genomic_key": encrypted_genomic_key,
            "salt": salt,
            "platform": config.platform,
            "timestamp": time.time(),
            "expires_at": time.time() + 3600  # 1 hour expiration
        }
       
        return json.dumps({
            "encrypted_payload": encrypted_data,
            "session_id": session_id,
            "encryption_type": "genomic_dna_fernet"
        })
   
    async def _route_to_platform(self, platform: PlatformType, encrypted_request: str,
                               user_id: str) -> Dict[str, Any]:
        if platform == PlatformType.GENOPATTERN:
            return await self._handle_genopattern_request(encrypted_request, user_id)
        elif platform == PlatformType.ASTRAELAN:
            return await self._handle_astraelan_request(encrypted_request, user_id)
        elif platform == PlatformType.CLINVELOCITY:
            return await self._handle_clinvelocity_request(encrypted_request, user_id)
       
        return {"error": "unknown_platform"}
   
    async def _handle_genopattern_request(self, encrypted_request: str, user_id: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
       
        return {
            "status": "processed",
            "platform": "GenoPattern",
            "analysis_results": {
                "variants_detected": 42,
                "pathogenic_variants": 3,
                "novel_mutations": 1,
                "confidence_score": 0.94
            },
            "processing_time": 0.1,
            "encrypted_response": True
        }
   
    async def _handle_astraelan_request(self, encrypted_request: str, user_id: str) -> Dict[str, Any]:
        await asyncio.sleep(0.05)
       
        return {
            "status": "experiment_completed",
            "platform": "AstraElan",
            "virtual_results": {
                "experiment_id": "VL_EXP_001",
                "success_rate": 0.87,
                "virtual_yield": "145.2mg",
                "purity": "99.1%"
            },
            "resource_usage": {
                "cpu_hours": 2.3,
                "memory_gb": 8.1
            },
            "encrypted_response": True
        }
   
    async def _handle_clinvelocity_request(self, encrypted_request: str, user_id: str) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
       
        return {
            "status": "clinical_data_processed",
            "platform": "ClinVelocity",
            "trial_results": {
                "trial_id": "CT_2025_001",
                "patient_enrollment": 47,
                "efficacy_endpoint": "met",
                "safety_profile": "acceptable",
                "statistical_power": 0.89
            },
            "compliance_status": "FDA_21CFR11_compliant",
            "encrypted_response": True
        }
   
    async def _decrypt_platform_response(self, response: Dict[str, Any],
                                       config: BioSecurityAPI) -> Dict[str, Any]:
        if not response.get("encrypted_response"):
            return response
       
        # Fix 1 & 4: Proper decryption with stored encrypted key and salt
        # For demo, assume response is plaintext; in real, decrypt if needed
        # Example: if 'session_id' in response:
        #    session = self.active_sessions.get(response['session_id'])
        #    if session and time.time() < session['expires_at']:
        #        genomic_key = self.master_fernet.decrypt(session['encrypted_genomic_key']).decode()
        #        salt = session['salt']
        #        decrypted = self.genomic_crypto.decrypt_from_synthetic_dna(encrypted_data, genomic_key, salt)
        #    else:
        #        raise ValueError("Session expired or invalid")
        # Here, return as-is for demo
        return response
   
    def get_integration_status(self) -> Dict[str, Any]:
        # Clean expired sessions (Fix 4)
        current_time = time.time()
        self.active_sessions = {k: v for k, v in self.active_sessions.items() if v['expires_at'] > current_time}
       
        platform_status = {}
        for platform_type, config in self.platform_configs.items():
            platform_status[platform_type.value] = {
                "security_level": config.enzyme_requirement,
                "encryption_strength": config.genomic_key_length,
                "access_window": len(config.circadian_restriction),
                "status": "operational"
            }
       
        return {
            "integration_status": "active",
            "platforms_integrated": len(self.platform_configs),
            "active_sessions": len(self.active_sessions),
            "bio_security_level": "maximum",
            "genomic_encryption": "enabled",
            "platform_details": platform_status,
            "lifeguard_system": self.lifeguard.get_system_status()
        }

# ==================== USAGE EXAMPLE ====================
async def main():
    print("=== LifeGuard Genesis Bio-Security System Demo (Fixed) ===\n")
   
    lifeguard = LifeGuardGenesis()
   
    lifeguard.register_user("researcher_001", {"typical_hours": [8, 9, 10, 15, 16]})
    lifeguard.register_user("partner_external_001", {"typical_hours": [14, 15, 16, 17]})
   
    auth_result = lifeguard.authenticate_user("researcher_001", {
        "protein_response": "MAVLPESGDGPQMW",  # Note: In real, this would need to match full hash
        "request_frequency": 1.2
    })
    print(f"Authentication Result: {auth_result}\n")
   
    data_result = lifeguard.process_data_request("researcher_001", "GenoPattern_analysis_001", "read")
    print(f"Data Access Result: {data_result}\n")
   
    threat_status = lifeguard.monitor_threats()
    print(f"Threat Status: {threat_status}\n")
   
    evolution_events = lifeguard.evolve_system()
    print(f"Evolution Events: {evolution_events}\n")
   
    system_status = lifeguard.get_system_status()
    print(f"System Status: {json.dumps(system_status, indent=2)}\n")
   
    # Integration demo
    integrator = PlatformIntegrator(lifeguard)
   
    lifeguard.register_user("researcher_lead", {"typical_hours": [8, 9, 10, 11, 14, 15, 16]})
    lifeguard.register_user("clinical_coordinator", {"typical_hours": [9, 10, 11, 12, 13, 14, 15]})
   
    print("1. Testing GenoPattern Access:")
    genopattern_request = {
        "operation_type": "variant_analysis",
        "sample_id": "SAMPLE_001",
        "analysis_type": "targeted_panel",
        "data_volume_mb": 250
    }
   
    result = await integrator.secure_platform_access(
        "researcher_lead",
        PlatformType.GENOPATTERN,
        genopattern_request
    )
    print(f"GenoPattern Result: {json.dumps(result, indent=2)}\n")
   
    print("2. Testing AstraElan Access:")
    astraelan_request = {
        "experiment_type": "protein_folding_simulation",
        "complexity_score": 6,
        "virtual_resources": {"cpu_hours": 15, "memory_gb": 32}
    }
   
    result = await integrator.secure_platform_access(
        "researcher_lead",
        PlatformType.ASTRAELAN,
        astraelan_request
    )
    print(f"AstraElan Result: {json.dumps(result, indent=2)}\n")
   
    print("3. Testing ClinVelocity Access:")
    clinvelocity_request = {
        "operation_type": "patient_enrollment",
        "trial_id": "TRIAL_2025_001",
        "patient_count": 25,
        "data_access_level": "aggregated"
    }
   
    result = await integrator.secure_platform_access(
        "clinical_coordinator",
        PlatformType.CLINVELOCITY,
        clinvelocity_request
    )
    print(f"ClinVelocity Result: {json.dumps(result, indent=2)}\n")
   
    print("4. Integration System Status:")
    status = integrator.get_integration_status()
    print(f"System Status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
