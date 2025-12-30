# Add this import near the top
from fastapi.responses import JSONResponse

# New endpoint for challenge
@app.get("/api/get-challenge/{user_id}")
async def get_challenge(user_id: str):
    requests_total.labels('/api/get-challenge').inc()
    result = lifeguard.get_protein_challenge(user_id)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

# Update authenticate to use the new flow
@app.post("/api/authenticate")
async def authenticate_user(auth_req: AuthenticationRequest = Body(...)):
    requests_total.labels('/api/authenticate').inc()
    success, message, details = lifeguard.authenticate_user(
        auth_req.user_id,
        {
            "protein_response": auth_req.protein_response,  # Now a full SHA-256 hash
            "request_frequency": auth_req.request_frequency
        }
    )
    if success:
        auth_success.inc()
    else:
        auth_failure.inc()
        if "quarantined" in message.lower():
            raise HTTPException(status_code=403, detail=message)
    return {"success": success, "message": message, "details": details}

# Remove any "security_flag": "honeypot_access" from data-request responses
# (already handled in new core.py)
