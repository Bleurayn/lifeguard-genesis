#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
from prometheus_client import Counter, Gauge, make_asgi_app, CollectorRegistry
import asyncio

# Import the fixed LifeGuardGenesis and PlatformIntegrator
# (Assume the fixed code is in lifeguard_genesis_fixed.py or inline it here)
from lifeguard_genesis_fixed import LifeGuardGenesis, PlatformIntegrator, PlatformType

app = FastAPI(title="LifeGuard Genesis API")

# Prometheus metrics
registry = CollectorRegistry()
requests_total = Counter('requests_total', 'Total API requests', ['endpoint'], registry=registry)
auth_success = Counter('auth_success', 'Successful authentications', registry=registry)
auth_failure = Counter('auth_failure', 'Failed authentications', registry=registry)
active_threats = Gauge('active_threats', 'Number of active threats', registry=registry)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app(registry=registry)
app.mount("/metrics", metrics_app)

# Initialize systems
lifeguard = LifeGuardGenesis()
integrator = PlatformIntegrator(lifeguard)

class UserRegistration(BaseModel):
    user_id: str
    typical_hours: List[int] = None

class AuthenticationRequest(BaseModel):
    user_id: str
    protein_response: str
    request_frequency: float = 1.0

class DataRequest(BaseModel):
    user_id: str
    resource_id: str
    request_type: str

class PlatformAccessRequest(BaseModel):
    user_id: str
    platform: str  # e.g., "genomic_analysis"
    request_data: Dict[str, Any]

@app.post("/api/register")
async def register_user(reg: UserRegistration = Body(...)):
    requests_total.labels('/api/register').inc()
    initial_biometrics = {"typical_hours": reg.typical_hours} if reg.typical_hours else None
    success = lifeguard.register_user(reg.user_id, initial_biometrics)
    if not success:
        raise HTTPException(status_code=400, detail="Registration failed")
    return {"status": "registered", "user_id": reg.user_id}

@app.post("/api/authenticate")
async def authenticate_user(auth_req: AuthenticationRequest = Body(...)):
    requests_total.labels('/api/authenticate').inc()
    success, message, details = lifeguard.authenticate_user(
        auth_req.user_id,
        {"protein_response": auth_req.protein_response, "request_frequency": auth_req.request_frequency}
    )
    if success:
        auth_success.inc()
    else:
        auth_failure.inc()
    return {"success": success, "message": message, "details": details}

@app.post("/api/data-request")
async def process_data_request(data_req: DataRequest = Body(...)):
    requests_total.labels('/api/data-request').inc()
    result = lifeguard.process_data_request(data_req.user_id, data_req.resource_id, data_req.request_type)
    if result["status"] == "error":
        raise HTTPException(status_code=429 if "rate limited" in result["message"].lower() else 400, detail=result["message"])
    return result

@app.get("/api/monitor-threats")
async def monitor_threats():
    requests_total.labels('/api/monitor-threats').inc()
    threats = lifeguard.monitor_threats()
    active_threats.set(threats["active_threats"])
    return threats

@app.get("/api/system-status")
async def get_system_status():
    requests_total.labels('/api/system-status').inc()
    return lifeguard.get_system_status()

@app.post("/api/platform-access")
async def secure_platform_access(access_req: PlatformAccessRequest = Body(...)):
    requests_total.labels('/api/platform-access').inc()
    try:
        platform = PlatformType(access_req.platform)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid platform type")
    result = await integrator.secure_platform_access(access_req.user_id, platform, access_req.request_data)
    if result["status"] != "success":
        raise HTTPException(status_code=403, detail=result.get("error", "Access denied"))
    return result

@app.get("/api/integration-status")
async def get_integration_status():
    requests_total.labels('/api/integration-status').inc()
    return integrator.get_integration_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
