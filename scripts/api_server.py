#!/usr/bin/env python3
"""
DAEDALUS v0.5 — Web API Server (api_server.py)

This server wraps Daedalus's core into a FastAPI application, allowing
interactivity via an HTTP API and serving a mobile-friendly frontend.

Usage:
    python scripts/api_server.py [--mock-mode] [--host 0.0.0.0] [--port 8000]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.secrets import load_secrets
from scripts.start_day import load_all_config, initialize_subsystems, load_local_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("daedalus.api")

app = FastAPI(title="Daedalus API", version="0.5")

# Mount static files (our web app folder)
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# Global subsystem states
app.state.engine = None
app.state.mock_mode = False

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Daedalus Backend...")
    load_secrets()
    
    # Check if mock mode is injected via command line
    # FastAPI doesn't easily pass args, so we rely on global state set before uvicorn.run
    if getattr(app.state, "mock_mode", False):
        logger.warning("MOCK MODE ACTIVE. LLM will NOT be loaded.")
        return
        
    config = load_all_config("config")
    subsystems = initialize_subsystems(config)
    
    logger.info("Loading Qwen model into VRAM... this may take a while.")
    model, tokenizer = load_local_model(config)
    
    from core.conversation import ConversationEngine
    engine = ConversationEngine(
        memory_store=subsystems["memory"],
        soul_bridge=subsystems["soul_bridge"],
        constitutional_core=subsystems["constitutional_core"],
        identity_manager=subsystems["identity"],
        entropy_scorer=subsystems["entropy_scorer"],
        config=config,
    )
    engine.set_local_model(model, tokenizer)
    subsystems["soul_bridge"].set_all_providers_mode("day")
    
    app.state.engine = engine
    logger.info("Daedalus engine is ready.")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = web_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Web directory not found. Please create /web/index.html</h1>"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    
    if getattr(app.state, "mock_mode", False):
        await asyncio.sleep(1) # simulate think time
        return {"response": f"*(Mock Mode)* You said: {user_input}. I am functioning perfectly."}
        
    if not app.state.engine:
        return JSONResponse(status_code=503, content={"detail": "Engine not initialized"})
        
    try:
        response = await app.state.engine.process_turn(user_input)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing turn: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/api/new")
async def new_conversation():
    if not getattr(app.state, "mock_mode", False) and app.state.engine:
        app.state.engine.new_conversation()
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAEDALUS Web Server")
    parser.add_argument("--mock-mode", action="store_true", help="Skip loading heavy models for testing")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()
    
    app.state.mock_mode = args.mock_mode
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
