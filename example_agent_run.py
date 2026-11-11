#!/usr/bin/env python3
import sys
import os
import time

# Ensure root of the project is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent_memory_manager import AgentMemoryManager

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def main():
    print_separator("TencentDB Agent Memory - Practical Simulation Scenario")

    # Initialize the core memory manager
    manager = AgentMemoryManager()

    user_id = "developer_alice"
    session_id = "session_42"

    # --- TURN 1: User declares a technical preference ---
    print_separator("Turn 1: User Declares Technical Preference")
    user_input_1 = "I prefer building serverless edge APIs using TypeScript and Cloudflare Workers with Wrangler."
    print(f"[User Alice]: {user_input_1}")

    # Save to memory (routes to both short-term buffer and long-term TencentDB vector store)
    print("\n[System]: Routing interaction to memory manager...")
    res_add = manager.add_memory(
        text=user_input_1,
        user_id=user_id,
        session_id=session_id,
        is_episodic=True,
        metadata={"category": "tech_preference", "role": "user"}
    )
    print(f"[System]: Saved memory successfully. ID: {res_add['id']} | Status: {res_add['status']}")

    # --- TURN 2: Dialogue turns continue (updates short-term cache) ---
    print_separator("Turn 2: Conversation continues (Short-term cache)")
    user_input_2 = "Can you also explain the difference between KV and Durable Objects?"
    print(f"[User Alice]: {user_input_2}")

    manager.add_memory(
        text=user_input_2,
        user_id=user_id,
        session_id=session_id,
        is_episodic=True,
        metadata={"category": "question", "role": "user"}
    )

    # Display short term memory buffer content
    print("\n[System]: Current Short-Term Memory Buffer for session:")
    for idx, turn in enumerate(manager.short_term_cache[session_id], 1):
        print(f"  {idx}. [{turn['role'].upper()}]: {turn['text']}")

    # --- TURN 3: Retrieval of Long-Term Memory (Context Injection) ---
    print_separator("Turn 3: Autonomous Retrieval & Context Injection")
    user_input_3 = "I want to start a new project today. What backend tech stack should I use?"
    print(f"[User Alice]: {user_input_3}")

    print("\n[System]: Querying TencentDB for relevant user preferences...")
    # Semantic search with recency-decay weighting
    relevant_memories = manager.search_relevant_memories(
        query_text="backend tech stack edge serverless preference",
        user_id=user_id,
        limit=3,
        apply_decay=True
    )

    print("\n[System]: Retrieved memories from TencentDB:")
    context_injected = ""
    for r in relevant_memories:
        print(f"  - [Score: {r['score']:.4f}] (Age: {r['elapsed_days'] * 86400:.2f}s ago) Text: \"{r['text']}\"")
        context_injected += f"- {r['text']}\n"

    print("\n[System]: Injecting retrieved context into the LLM Prompt...")
    system_prompt = f"""
    You are an expert AI Assistant.
    Below is relevant context retrieved from Alice's long-term memory:
    {context_injected}
    Answer Alice's question considering her preferences.
    """
    print("-" * 80)
    print(system_prompt.strip())
    print("-" * 80)

    # Simulate LLM Response using memory content
    simulated_agent_response = "Based on your preferences, you should definitely use TypeScript with Cloudflare Workers and wrangler, as you prefer building serverless edge APIs."
    print(f"[AI Agent Response]: {simulated_agent_response}")

    # Save agent turn to short-term cache
    manager.add_memory(
        text=simulated_agent_response,
        user_id=user_id,
        session_id=session_id,
        is_episodic=False,
        metadata={"role": "assistant"}
    )

    # --- TURN 4: Summarization & Consolidation ---
    print_separator("Turn 4: Session Summarization & Consolidation")
    print("[System]: Consolidating short-term history...")
    summary_res = manager.summarize_session(session_id=session_id, user_id=user_id)
    print(f"[System]: Summary created and saved to TencentDB!")
    print(f"  - Summary Text: {summary_res['summary_text']}")
    print(f"  - Turn Count Consolidated: {summary_res['turn_count']}")

    # --- TURN 5: Forgetting & Purge Cycle ---
    print_separator("Turn 5: Forgetting Cycle & GDPR Purge")
    print("[System]: Executing forgetting cycle with high min_relevance threshold (simulating high decay)...")
    forget_res = manager.forget_outdated(user_id=user_id, min_relevance=0.99)
    print(f"[System]: Purged memories count: {forget_res['purged_count']}")
    print(f"  - Purged IDs: {forget_res['purged_ids']}")

    print("\n[System]: Executing GDPR/LGPD compliance purge to wipe user's identity data...")
    purge_success = manager.clear_all_user_memories(user_id=user_id)
    print(f"[System]: GDPR Purge execution status: {'SUCCESS' if purge_success else 'FAILED'}")

    print_separator("Simulation Completed Successfully!")

if __name__ == "__main__":
    main()
