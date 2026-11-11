import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from src.config import settings
from src.tencent_vector_connector import TencentVectorConnector
from src.agent_memory_manager import AgentMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TencentDB Agent Memory Platform",
    description="Enterprise-grade dynamic, structured, and long-term memory engine for AI Agents built on Tencent Cloud VectorDB.",
    version="1.0.0"
)

# Initialize global memory manager singleton
memory_manager = AgentMemoryManager()

# --- Request & Response Models ---

class AddMemoryRequest(BaseModel):
    text: str = Field(..., description="The content of the dialogue turn or fact to remember.")
    user_id: str = Field(..., description="Unique identifier of the end user.")
    session_id: str = Field(..., description="Unique session/dialogue identifier.")
    is_episodic: bool = Field(True, description="Whether to store this memory in the long-term Vector DB.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Arbitrary metadata including role, entities, etc.")

class AddMemoryResponse(BaseModel):
    id: str
    status: str
    deduplicated: bool
    timestamp: float

class MemoryItem(BaseModel):
    id: str
    text: str
    score: float
    user_id: str
    session_id: str
    timestamp: float
    original_score: float
    decay_factor: float
    elapsed_days: float
    metadata: Dict[str, Any]

class QueryMemoriesResponse(BaseModel):
    query: str
    results: List[MemoryItem]

class SummarizeRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier to summarize.")
    user_id: str = Field(..., description="Unique user ID associated with the session.")

class SummarizeResponse(BaseModel):
    status: str
    summary_id: Optional[str] = None
    summary_text: Optional[str] = None
    turn_count: int

class ForgetRequest(BaseModel):
    user_id: str = Field(..., description="User ID to execute forgetting purge on.")
    min_relevance: float = Field(0.3, description="Minimum relevance score threshold. Memories with lower scores will be forgotten.")

class ForgetResponse(BaseModel):
    purged_count: int
    retained_count: int
    purged_ids: List[str]

# --- Endpoints ---

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns the operational status of the memory engine."""
    return {
        "status": "healthy",
        "mock_mode": memory_manager.connector.mock_mode,
        "database": settings.TENCENTDB_DATABASE_NAME,
        "collection": settings.TENCENTDB_COLLECTION_NAME,
        "sdk_loaded": TencentVectorConnector is not None
    }

@app.post("/memories", response_model=AddMemoryResponse, status_code=status.HTTP_201_CREATED, tags=["Memory Core"])
def add_new_memory(req: AddMemoryRequest):
    """Adds a dialogue turn or key fact into the agent memory layers (short-term buffer & long-term vector store)."""
    try:
        res = memory_manager.add_memory(
            text=req.text,
            user_id=req.user_id,
            session_id=req.session_id,
            is_episodic=req.is_episodic,
            metadata=req.metadata
        )
        if res.get("status") == "failed":
            raise HTTPException(status_code=500, detail="Failed to persist memory to TencentDB.")
        return AddMemoryResponse(
            id=res["id"],
            status=res["status"],
            deduplicated=res.get("deduplicated", False),
            timestamp=res["timestamp"]
        )
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", response_model=QueryMemoriesResponse, tags=["Memory Core"])
def query_relevant_memories(
    query: str = Query(..., description="The context or statement to search against."),
    user_id: str = Query(..., description="The user identifier."),
    session_id: Optional[str] = Query(None, description="Optional session filter."),
    limit: int = Query(5, description="Maximum number of relevant records to retrieve."),
    apply_decay: bool = Query(True, description="Whether to apply the exponential decay formula based on elapsed days.")
):
    """Queries TencentDB for semantically similar memories, incorporating recency-decay weighting."""
    try:
        results = memory_manager.search_relevant_memories(
            query_text=query,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            apply_decay=apply_decay
        )

        items = []
        for r in results:
            items.append(MemoryItem(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                user_id=r["user_id"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                original_score=r["original_score"],
                decay_factor=r["decay_factor"],
                elapsed_days=r["elapsed_days"],
                metadata=r["metadata"]
            ))

        return QueryMemoriesResponse(query=query, results=items)
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/summarize", response_model=SummarizeResponse, tags=["Memory Consolidation"])
def consolidate_session_history(req: SummarizeRequest):
    """Consolidates short-term session conversation history into a persistent long-term summary inside TencentDB."""
    try:
        res = memory_manager.summarize_session(session_id=req.session_id, user_id=req.user_id)
        return SummarizeResponse(
            status=res["status"],
            summary_id=res.get("summary_id"),
            summary_text=res.get("summary_text"),
            turn_count=res["turn_count"]
        )
    except Exception as e:
        logger.error(f"Error consolidating session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/forget", response_model=ForgetResponse, tags=["Memory Consolidation"])
def execute_forgetting_cycle(req: ForgetRequest):
    """Executes the forgetting/purge cycle, removing memories that fell below the minimum relevance threshold."""
    try:
        res = memory_manager.forget_outdated(user_id=req.user_id, min_relevance=req.min_relevance)
        return ForgetResponse(
            purged_count=res["purged_count"],
            retained_count=res["retained_count"],
            purged_ids=res["purged_ids"]
        )
    except Exception as e:
        logger.error(f"Error executing forgetting cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/user/{user_id}", tags=["GDPR Compliance"])
def purge_user_identity_memories(user_id: str) -> Dict[str, Any]:
    """GDPR/LGPD compliance endpoint to wipe all personal long-term memories belonging to a user."""
    try:
        success = memory_manager.clear_all_user_memories(user_id=user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to purge user memories from database.")
        return {
            "status": "success",
            "message": f"All memories for user {user_id} have been permanently deleted."
        }
    except Exception as e:
        logger.error(f"Error purging user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
