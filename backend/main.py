"""
FastAPI server for the Vulcan OmniPro 220 agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path
import anthropic
from openai import OpenAI

from advanced_agent import AdvancedVulcanAgent as VulcanAgent
from vector_store import VectorStore, build_vector_store_from_knowledge_base
from knowledge_extractor import KnowledgeExtractor

app = FastAPI(title="Vulcan OmniPro 220 Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend statically (for single-service deploy on Railway/Koyeb)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

# Global agent instance
agent: Optional[VulcanAgent] = None
vector_store: Optional[VectorStore] = None
knowledge_base: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    artifacts: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    usage: Dict[str, int]
    technical_response: Optional[Dict[str, Any]] = None


class WeldAnalysisRequest(BaseModel):
    image: str  # base64 encoded image


class WeldAnalysisResponse(BaseModel):
    defects: List[str]
    severity: str  # 'low', 'medium', 'high'
    causes: List[str]
    solutions: List[str]
    settingsRecommendation: Optional[Dict[str, str]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent, vector_store, knowledge_base

    print("Initializing Vulcan OmniPro 220 Agent...")

    # Check if knowledge base exists
    kb_file = Path("knowledge_base.json")

    if not kb_file.exists():
        print("Knowledge base not found. Building from PDFs...")
        extractor = KnowledgeExtractor()
        knowledge_base = extractor.process_all_manuals()

        # Also load the full KB with images
        knowledge_base = extractor.get_knowledge_base()
    else:
        print("Loading existing knowledge base...")
        with open(kb_file, 'r') as f:
            kb_lite = json.load(f)

        # Re-extract to get full images
        extractor = KnowledgeExtractor()
        knowledge_base = extractor.process_all_manuals()

    # Build vector store
    print("Building vector store...")
    vector_store = build_vector_store_from_knowledge_base(knowledge_base)

    # Initialize agent
    print("Initializing agent...")
    agent = VulcanAgent(vector_store, knowledge_base)

    print("✓ Agent ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Vulcan OmniPro 220 Agent API",
        "knowledge_base_stats": {
            "text_chunks": len(knowledge_base.get("text_chunks", [])),
            "images": len(knowledge_base.get("images", [])),
            "knowledge_nodes": len(knowledge_base.get("knowledge_nodes", [])),
            "tables": len(knowledge_base.get("tables", [])),
            "procedures": len(knowledge_base.get("procedures", [])),
            "relationships": len(knowledge_base.get("relationships", [])),
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for the agent.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        response = agent.chat(
            user_message=request.message,
            image_data=request.image_data
        )

        return ChatResponse(
            text=response["text"],
            artifacts=response.get("artifacts", []),
            images=response.get("images", []),
            usage=response.get("usage", {}),
            technical_response=response.get("technical_response"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_conversation():
    """Reset the conversation history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    agent.reset_conversation()
    return {"status": "ok", "message": "Conversation reset"}


@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base."""
    return {
        "text_chunks": len(knowledge_base.get("text_chunks", [])),
        "images": len(knowledge_base.get("images", [])),
        "tables": len(knowledge_base.get("tables", [])),
        "procedures": len(knowledge_base.get("procedures", [])),
        "knowledge_nodes": len(knowledge_base.get("knowledge_nodes", [])),
        "relationships": len(knowledge_base.get("relationships", [])),
        "vector_store_count": vector_store.get_collection_count() if vector_store else 0,
    }


class ApiKeyRequest(BaseModel):
    key: str
    provider: str = "openrouter"  # "openrouter" or "anthropic"


class ApiKeyStatus(BaseModel):
    configured: bool
    provider: str


@app.get("/api-key")
async def get_api_key_status():
    """Check if an API key is configured."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    provider = "none"
    configured = False

    if openrouter_key and openrouter_key != "your-api-key-here":
        provider = "openrouter"
        configured = True
    elif anthropic_key and anthropic_key != "your-api-key-here":
        provider = "anthropic"
        configured = True

    return ApiKeyStatus(configured=configured, provider=provider)


@app.post("/api-key")
async def set_api_key(request: ApiKeyRequest):
    """Set API key at runtime. Reloads the agent with the new key."""
    global agent

    if not request.key or not request.key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")

    # Validate the key format
    if request.provider == "openrouter":
        if not request.key.startswith("sk-or-"):
            raise HTTPException(status_code=400, detail="OpenRouter keys start with 'sk-or-'")
        os.environ["OPENROUTER_API_KEY"] = request.key.strip()
    elif request.provider == "anthropic":
        if not request.key.startswith("sk-ant-"):
            raise HTTPException(status_code=400, detail="Anthropic keys start with 'sk-ant-'")
        os.environ["ANTHROPIC_API_KEY"] = request.key.strip()
    else:
        raise HTTPException(status_code=400, detail="Provider must be 'openrouter' or 'anthropic'")

    # Reload the agent with the new key
    try:
        agent = VulcanAgent(vector_store, knowledge_base)
        return {"status": "ok", "message": f"API key set for {request.provider}", "provider": request.provider}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {str(e)}")


@app.post("/analyze-weld", response_model=WeldAnalysisResponse)
async def analyze_weld(request: WeldAnalysisRequest):
    """
    Analyze a weld image for defects using Claude Vision.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        response = agent.analyze_weld_defect(image_base64=request.image)

        return WeldAnalysisResponse(
            defects=response["defects"],
            severity=response["severity"],
            causes=response["causes"],
            solutions=response["solutions"],
            settingsRecommendation=response.get("settingsRecommendation")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
