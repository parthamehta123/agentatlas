"""Canonical registry data model.

A *candidate* is a raw observation from one discovery adapter ("something in k8s that
looks like it calls an LLM"). Identity resolution groups candidates into one *logical
agent*; replicas and multi-cluster deployments become ``instances`` of that agent, and
a single multi-agent host becomes several agents.

Lifecycle is deliberately graded - discovering an agent never makes it trusted:

    CANDIDATE -> DISCOVERED -> VERIFIED -> REACHABLE -> AUTHORIZED
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Status(IntEnum):
    CANDIDATE = 0  # observed, not (yet) believed to be an agent
    DISCOVERED = 1  # believed to be an agent
    VERIFIED = 2  # corroborated by independent evidence families
    REACHABLE = 3  # has a known invocation contract (endpoint + protocol)
    AUTHORIZED = 4  # policy allows the platform to execute it


class Kind(StrEnum):
    AGENT = "agent"
    LLM_WORKLOAD = "llm_workload"  # calls an LLM; agentic-ness unknown
    LLM_CLIENT = "llm_client"  # calls an LLM, never uses tools (batch jobs etc.)
    MCP_SERVER = "mcp_server"  # exposes tools; not an agent
    SERVICE = "service"


class Family(StrEnum):
    """Independent kinds of evidence. Verification counts families, not sources."""

    DECLARED = "declared"  # registration, A2A card, explicit annotation
    RUNTIME = "runtime"  # k8s / cloud inventory
    BEHAVIORAL = "behavioral"  # OTel agent spans, tool-calling LLM traffic
    OWNERSHIP = "ownership"  # CMDB / catalog


class Candidate(BaseModel):
    source: str
    source_ref: str
    kind: Kind
    families: set[Family]
    identity_keys: set[str]
    attributes: dict[str, Any] = Field(default_factory=dict)
    replicas: int = 0


class Evidence(BaseModel):
    source: str
    source_ref: str
    families: list[Family]


class CategoryStats(BaseModel):
    runs: int = 0
    successes: int = 0
    errors: int = 0
    p95_latency_ms: float | None = None
    cost_usd_total: float = 0.0
    retries: int = 0
    guardrail_violations: int = 0


class AgentRecord(BaseModel):
    agent_id: str
    slug: str
    name: str
    kind: Kind = Kind.AGENT
    status: Status = Status.CANDIDATE
    inferred: bool = False  # classified as an agent purely from behaviour
    description: str = ""
    owner_team: str | None = None
    department: str | None = None
    framework: str | None = None
    model: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    inferred_capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    data_clearance: str | None = None
    endpoint: str | None = None
    protocol: str | None = None  # a2a | http
    instances: int = 0
    evidence: list[Evidence] = Field(default_factory=list)
    telemetry: dict[str, CategoryStats] = Field(default_factory=dict)
    tools_used: dict[str, int] = Field(default_factory=dict)
    llm_requests: int = 0
    policy_notes: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {e.source for e in self.evidence}

    @property
    def families(self) -> set[Family]:
        return {f for e in self.evidence for f in e.families}

    def all_capabilities(self) -> list[str]:
        return list(dict.fromkeys(self.capabilities + self.inferred_capabilities))

    def total_runs(self) -> int:
        return sum(s.runs for s in self.telemetry.values())
