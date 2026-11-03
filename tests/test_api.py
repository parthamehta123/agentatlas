import importlib

from fastapi.testclient import TestClient


def test_api_flow(data_dir, monkeypatch):
    monkeypatch.setenv("AGENTATLAS_INFRA", str(data_dir / "infra"))
    from agentatlas.api import main

    main = importlib.reload(main)
    c = TestClient(main.app)

    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/agents/register", json={
        "agent_id": "agt-new-001", "name": "Brand New Agent", "owner_team": "ml-team",
        "capabilities": ["sql"], "endpoint": "https://new.internal", "protocol": "http"})
    assert r.status_code == 202
    assert c.post("/agents/register", json={"agent_id": "x", "name": "n",
                                            "owner_team": "t"}).status_code == 422

    disc = c.post("/discovery/run", json={"include_a2a": False}).json()
    assert disc["agents"] > 1000
    assert c.get("/agents/agt-new-001").json()["status"] == "DISCOVERED"  # one source only

    assert c.post("/ml/build").json()["clusters"] == 12
    assert len(c.get("/clusters").json()) == 12
    sel = c.post("/select/task", json={"description": "reconcile the cash ledger",
                                       "required_capabilities": ["reconciliation"],
                                       "data_classification": "confidential", "k": 5}).json()
    assert 0 < len(sel["agent_ids"]) <= 5 and sel["funnel"]
    assert len(c.post("/select/security", json={"k": 7}).json()["agent_ids"]) == 7
    assert c.get("/agents/does-not-exist").status_code == 404
