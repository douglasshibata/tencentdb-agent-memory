import logging
import uuid
import time
import math
from typing import List, Dict, Any, Optional
from src.config import settings
from src.tencent_vector_connector import TencentVectorConnector

logger = logging.getLogger(__name__)

class AgentMemoryManager:
    """
    Core memory management engine for AI Agents.
    Manages short-term buffer cache, long-term semantic persistence,
    deduplication, recency weighting (decay), summarization, and forgetting.
    """
    def __init__(self):
        self.connector = TencentVectorConnector()
        # Short term cache structure: {session_id: [list of dialogue turns]}
        self.short_term_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.short_term_limit = settings.MEMORY_SHORT_TERM_LIMIT
        self.deduplication_threshold = settings.MEMORY_DEDUPLICATION_THRESHOLD
        self.decay_rate = settings.MEMORY_DECAY_RATE

    def add_memory(
        self,
        text: str,
        user_id: str,
        session_id: str,
        is_episodic: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Routes and persists a dialogue turn or key fact into memory.
        Updates short-term cache and triggers semantic deduplication check before
        writing to long-term TencentDB Vector store.
        """
        now = time.time()
        m_id = str(uuid.uuid4())
        meta = metadata or {}

        # 1. Update Short-Term Memory Buffer
        if session_id not in self.short_term_cache:
            self.short_term_cache[session_id] = []

        dialogue_turn = {
            "id": m_id,
            "text": text,
            "timestamp": now,
            "role": meta.get("role", "user")
        }
        self.short_term_cache[session_id].append(dialogue_turn)

        # Keep short term buffer within bounds
        if len(self.short_term_cache[session_id]) > self.short_term_limit:
            self.short_term_cache[session_id].pop(0)

        # 2. Handle Long-Term Episodic / Semantic Memory Persistence
        if is_episodic:
            # Check for duplicates (Deduplication Mechanism)
            duplicates = self.connector.search_similarity(
                query_text=text,
                user_id=user_id,
                session_id=session_id,
                limit=1
            )

            if duplicates and duplicates[0]["score"] >= self.deduplication_threshold:
                # Deduplicate: Update existing memory with fresh timestamp/metadata instead of inserting a duplicate
                existing_id = duplicates[0]["id"]
                logger.info(f"Deduplication triggered. Updating existing memory {existing_id} (score {duplicates[0]['score']:.4f})")
                m_id = existing_id
                meta["update_count"] = meta.get("update_count", 0) + 1

            # Persist to Tencent DB Vector Store
            success = self.connector.add_document(
                id=m_id,
                text=text,
                user_id=user_id,
                session_id=session_id,
                timestamp=now,
                metadata=meta
            )
            return {
                "id": m_id,
                "status": "persisted" if success else "failed",
                "deduplicated": m_id == duplicates[0]["id"] if (duplicates and duplicates[0]["score"] >= self.deduplication_threshold) else False,
                "timestamp": now
            }

        return {
            "id": m_id,
            "status": "cached_only",
            "timestamp": now
        }

    def search_relevant_memories(
        self,
        query_text: str,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        apply_decay: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant memories from TencentDB, applying Cosine Similarity
        combined with Recency Weighting (Exponential Decay).
        Final Relevance Score = CosineSimilarity * e^(-lambda * t_days)
        """
        # Fetch semantic candidates from TencentDB
        candidates = self.connector.search_similarity(
            query_text=query_text,
            user_id=user_id,
            session_id=session_id,
            limit=limit * 2 # fetch extra candidates to sort after decay calculation
        )

        now = time.time()
        scored_memories = []

        for cand in candidates:
            orig_score = cand["score"]
            timestamp = cand["timestamp"]

            # Calculate elapsed days since memory was created
            elapsed_seconds = max(0.0, now - timestamp)
            elapsed_days = elapsed_seconds / 86400.0  # 1 day = 86400 seconds

            # Calculate Exponential decay: S_final = S_orig * e^(-lambda * t)
            decay_factor = math.exp(-self.decay_rate * elapsed_days)
            final_score = orig_score * decay_factor if apply_decay else orig_score

            cand["original_score"] = orig_score
            cand["decay_factor"] = decay_factor
            cand["score"] = final_score
            cand["elapsed_days"] = elapsed_days

            scored_memories.append(cand)

        # Re-sort candidates based on decay-weighted scores
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return scored_memories[:limit]

    def summarize_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Summarization & Consolidation Mechanism.
        Consolidates short-term conversation buffer into a long-term semantic summary.
        Saves the summary as a persistent fact in TencentDB.
        """
        buffer = self.short_term_cache.get(session_id, [])
        if not buffer:
            return {"status": "empty", "message": "No conversation turns to summarize."}

        # Standard analytical template for mock/simulation summarization
        # In a real environment, this prompt would go to an LLM
        exchanges = []
        for turn in buffer:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            exchanges.append(f"{role.capitalize()}: {text}")

        conversation_transcript = "\n".join(exchanges)

        # Simulate fact extraction and summarization
        summary_text = f"Resumo Consolidado da Sessão {session_id}: O usuário discutiu tópicos técnicos e preferências do projeto, incluindo requisitos do TencentDB Agent Memory. Transcrição consolidada de {len(buffer)} interações."

        # Persist summary to long-term memory
        m_id = str(uuid.uuid4())
        success = self.connector.add_document(
            id=m_id,
            text=summary_text,
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            metadata={"type": "session_summary", "turn_count": len(buffer)}
        )

        return {
            "status": "consolidated" if success else "failed",
            "summary_id": m_id,
            "summary_text": summary_text,
            "turn_count": len(buffer)
        }

    def forget_outdated(self, user_id: str, min_relevance: float = 0.3) -> Dict[str, Any]:
        """
        Forgetting Mechanism.
        Iterates over user's long-term memories and purges those whose
        decayed scores fall below a minimum relevance threshold.
        """
        # Fetch all records for the user using a dummy search query
        # Since we use similarity search, we can retrieve with a broad query
        # In a real system, you might list documents, but with our vector store
        # we can fetch up to 100 items to evaluate relevance decay.
        all_memories = self.connector.search_similarity(
            query_text="memory fact preference info",
            user_id=user_id,
            limit=100
        )

        now = time.time()
        purged_count = 0
        retained_count = 0
        purged_ids = []

        for mem in all_memories:
            timestamp = mem["timestamp"]
            elapsed_days = max(0.0, now - timestamp) / 86400.0
            decay_factor = math.exp(-self.decay_rate * elapsed_days)
            decayed_score = mem["score"] * decay_factor

            if decayed_score < min_relevance:
                # Forget: delete from Vector DB
                self.connector.delete_document(mem["id"])
                purged_ids.append(mem["id"])
                purged_count += 1
            else:
                retained_count += 1

        return {
            "purged_count": purged_count,
            "retained_count": retained_count,
            "purged_ids": purged_ids
        }

    def clear_short_term(self, session_id: str):
        """Clears short-term buffer for a given session."""
        if session_id in self.short_term_cache:
            del self.short_term_cache[session_id]

    def clear_all_user_memories(self, user_id: str) -> bool:
        """Deletes all persistent memories for a user."""
        return self.connector.delete_by_user(user_id)
