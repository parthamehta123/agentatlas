"""Tiny self-registration client - the non-A2A way for an agent to announce itself.

    from agentatlas.sdk import register
    register("https://atlas.internal", agent_id="agt-fraud-7", name="Fraud Scout",
             owner_team="fraud-team-2", capabilities=["fraud_detection"],
             endpoint="https://fraud-scout.internal/invoke", protocol="http")

Call it from the agent's startup hook. It is idempotent (same agent_id = update).
"""

from __future__ import annotations

import httpx


def register(atlas_url: str, *, timeout: float = 5.0, **payload) -> dict:
    resp = httpx.post(atlas_url.rstrip("/") + "/agents/register", json=payload,
                      timeout=timeout)
    resp.raise_for_status()
    return resp.json()
