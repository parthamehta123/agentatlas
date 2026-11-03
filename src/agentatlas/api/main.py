"""FastAPI service: ``uvicorn agentatlas.api.main:app``.

Discovery reads from ``AGENTATLAS_INFRA`` (a server-side directory); clients cannot
point it at arbitrary paths.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentatlas.registry.models import Status
from agentatlas.selection.engine import TaskConfig, TaskSpec
from agentatlas.service import Atlas

app = FastAPI(title="AgentAtlas", version="0.1.0",
              description="Protocol-agnostic discovery, clustering and selection of AI agents")
atlas = Atlas(os.environ.get("AGENTATLAS_DB", ":memory:"))
INFRA = Path(os.environ.get("AGENTATLAS_INFRA", "data/infra"))


class RegistrationIn(BaseModel):
    agent_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{3,64}$")
    name: str
    owner_team: str
    department: str | None = None
    description: str = ""
    framework: str | None = None
    model: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    data_clearance: str | None = None
    endpoint: str | None = None
    protocol: str | None = None


class DiscoveryIn(BaseModel):
    include_a2a: bool = True


class TaskIn(BaseModel):
    description: str
    category: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    data_classification: str = "internal"
    max_p95_latency_ms: float | None = None
    max_cost_per_task: float | None = None
    k: int = Field(10, ge=1, le=50)
    mmr_lambda: float = Field(0.75, ge=0, le=1)
    cluster_probe: int | None = Field(None, ge=1)


class KIn(BaseModel):
    k: int = Field(10, ge=1, le=50)
    outlier_slots: int = Field(1, ge=0, le=5)


def _selection(sel) -> dict:
    out = asdict(sel)
    out["meta"].pop("retrieved", None)
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "registry": atlas.registry.counts(),
            "model_built": atlas.selector is not None}


@app.post("/agents/register", status_code=202)
def register(body: RegistrationIn) -> dict:
    atlas.register(body.model_dump())
    return {"accepted": body.agent_id, "note": "reconciled on next /discovery/run"}


@app.post("/discovery/run")
def discovery(body: DiscoveryIn) -> dict:
    rep = atlas.discover(INFRA if INFRA.exists() else None, body.include_a2a)
    out = asdict(rep)
    out.pop("records")
    out["conflicts"] = out["conflicts"][:20]
    return out


@app.get("/agents")
def list_agents(min_status: Status = Status.DISCOVERED, limit: int = 100,
                offset: int = 0) -> list[dict]:
    rows = atlas.registry.agents(min_status)[offset:offset + limit]
    return [{"agent_id": r.agent_id, "name": r.name, "status": r.status.name,
             "capabilities": r.all_capabilities(), "sources": sorted(r.sources)} for r in rows]


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str) -> dict:
    rec = atlas.registry.get(agent_id)
    if rec is None:
        raise HTTPException(404, "unknown agent")
    return rec.model_dump(mode="json") | {"status": rec.status.name}


@app.post("/ml/build")
def build(k: int = 12) -> dict:
    try:
        fs = atlas.build(k)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {"agents": len(fs.records), "clusters": atlas.clusters.n_clusters}


@app.get("/clusters")
def clusters() -> list[dict]:
    return atlas.cluster_summary()


@app.post("/select/task")
def select_task(body: TaskIn) -> dict:
    spec = TaskSpec(**body.model_dump(exclude={"k", "mmr_lambda", "cluster_probe"}))
    cfg = TaskConfig(k=body.k, mmr_lambda=body.mmr_lambda, cluster_probe=body.cluster_probe)
    return _selection(atlas.select_task(spec, cfg))


@app.post("/select/representative")
def select_representative(body: KIn) -> dict:
    return _selection(atlas.select_representative(body.k, body.outlier_slots))


@app.post("/select/security")
def select_security(body: KIn) -> dict:
    return _selection(atlas.select_security(body.k))
