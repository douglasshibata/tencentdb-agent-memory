import pytest
import time
from fastapi.testclient import TestClient
from src.config import settings
from src.tencent_vector_connector import TencentVectorConnector, EmbeddingGenerator
from src.agent_memory_manager import AgentMemoryManager
from src.api import app

client = TestClient(app)

def test_settings_load():
    """Verify settings parse variables properly with default fallbacks."""
    assert settings.TENCENTDB_HOST is not None
    assert settings.MEMORY_SHORT_TERM_LIMIT > 0
    assert settings.MEMORY_DEDUPLICATION_THRESHOLD > 0

def test_embedding_generator():
    """Test unit length vector generation."""
    text = "TencentDB Agent Memory is amazing!"
    vector = EmbeddingGenerator.generate(text)
    assert len(vector) == 1536
    # Unit vector magnitude check: sum(x^2) should be very close to 1.0
    magnitude = sum(x * x for x in vector)
    assert abs(magnitude - 1.0) < 1e-5

def test_vector_connector_operations():
    """Test standard connector CRUD operations."""
    connector = TencentVectorConnector()
    assert connector.mock_mode is True or connector.mock_mode is False

    doc_id = "test_doc_99"
    user_id = "user_test_99"
    session_id = "session_test_99"
    text = "I love Python and Go."

    # Clean up before testing
    connector.delete_document(doc_id)

    # Create / Add
    success = connector.add_document(
        id=doc_id,
        text=text,
        user_id=user_id,
        session_id=session_id,
        timestamp=time.time(),
        metadata={"lang": "python_go"}
    )
    assert success is True

    # Search
    search_res = connector.search_similarity(
        query_text="programming languages Python",
        user_id=user_id,
        session_id=session_id,
        limit=2
    )
    assert len(search_res) > 0
    assert search_res[0]["id"] == doc_id
    assert "Python" in search_res[0]["text"]

    # Clean up
    del_success = connector.delete_document(doc_id)
    assert del_success is True

def test_memory_manager_deduplication():
    """Test that highly similar or identical memories are deduplicated during write."""
    manager = AgentMemoryManager()
    user_id = "alice_dedup"
    session_id = "session_dedup"

    # Clear any leftover memories
    manager.clear_all_user_memories(user_id)

    # Insert first memory
    text_1 = "My favorite cloud database is TencentDB."
    res_1 = manager.add_memory(text_1, user_id, session_id)
    id_1 = res_1["id"]

    # Insert second highly similar memory (should trigger deduplication and reuse the ID)
    text_2 = "My absolute favorite cloud database is TencentDB."
    res_2 = manager.add_memory(text_2, user_id, session_id)
    id_2 = res_2["id"]

    # Deduplication check
    assert id_1 == id_2
    assert res_2["deduplicated"] is True

def test_memory_manager_decay_recency():
    """Test that search weights apply decay properly to older memories."""
    manager = AgentMemoryManager()
    user_id = "bob_decay"
    session_id = "session_decay"

    manager.clear_all_user_memories(user_id)

    # Add an old memory (backdated)
    old_time = time.time() - (10 * 86400)  # 10 days ago
    doc_id_old = "old_memory_fact"
    manager.connector.add_document(
        id=doc_id_old,
        text="I bought a new car ten days ago.",
        user_id=user_id,
        session_id=session_id,
        timestamp=old_time,
        metadata={"type": "purchase"}
    )

    # Add a fresh memory (now)
    doc_id_fresh = "fresh_memory_fact"
    manager.connector.add_document(
        id=doc_id_fresh,
        text="I bought a new car today.",
        user_id=user_id,
        session_id=session_id,
        timestamp=time.time(),
        metadata={"type": "purchase"}
    )

    # Query with decay applied vs query without decay
    results_decayed = manager.search_relevant_memories(
        query_text="bought a new car",
        user_id=user_id,
        limit=2,
        apply_decay=True
    )

    results_undecayed = manager.search_relevant_memories(
        query_text="bought a new car",
        user_id=user_id,
        limit=2,
        apply_decay=False
    )

    # The fresh memory must have a higher relative score in the decayed list
    # because the old one lost relevance weight due to time decay!
    assert len(results_decayed) == 2
    assert results_decayed[0]["id"] == "fresh_memory_fact"

def test_api_endpoints():
    """Integration test checking FastAPI endpoints."""
    # 1. Health check
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # 2. Add Memory
    user_id = "api_user_test"
    session_id = "api_session_test"
    payload = {
        "text": "I absolutely love FastAPI web framework development.",
        "user_id": user_id,
        "session_id": session_id,
        "is_episodic": True,
        "metadata": {"role": "user", "tag": "web"}
    }
    resp_add = client.post("/memories", json=payload)
    assert resp_add.status_code == 201
    add_data = resp_add.json()
    assert "id" in add_data
    assert add_data["status"] == "persisted"

    # 3. Query memories
    resp_query = client.get(f"/memories?query=FastAPI development&user_id={user_id}")
    assert resp_query.status_code == 200
    query_data = resp_query.json()
    assert len(query_data["results"]) > 0
    assert "FastAPI" in query_data["results"][0]["text"]

    # 4. Summarize session history
    resp_sum = client.post("/memories/summarize", json={"session_id": session_id, "user_id": user_id})
    assert resp_sum.status_code == 200
    sum_data = resp_sum.json()
    assert sum_data["status"] == "consolidated"
    assert "Resumo Consolidado" in sum_data["summary_text"]

    # 5. Forgetting purge cycle
    resp_forget = client.post("/memories/forget", json={"user_id": user_id, "min_relevance": 0.99})
    assert resp_forget.status_code == 200
    forget_data = resp_forget.json()
    assert "purged_count" in forget_data

    # 6. Delete user memories GDPR
    resp_del = client.delete(f"/memories/user/{user_id}")
    assert resp_del.status_code == 200
    assert "permanently deleted" in resp_del.json()["message"]
