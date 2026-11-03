"""Verification and authorisation: graded trust, never implied by discovery."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentatlas.registry.models import AgentRecord, Family, Kind, Status
from agentatlas.tool_catalog import BLOCKED_TOOLS


@dataclass
class Policy:
    blocked_tools: set[str] = field(default_factory=lambda: set(BLOCKED_TOOLS))
    max_guardrail_violation_rate: float = 0.01
    min_runs_for_violation_check: int = 20
    require_owner: bool = True
    require_clearance: bool = True


def assess(rec: AgentRecord, policy: Policy) -> AgentRecord:
    """Set ``status`` and ``policy_notes`` in place and return the record."""
    notes: list[str] = []
    if rec.kind != Kind.AGENT:
        rec.status = Status.CANDIDATE
        rec.policy_notes = [f"classified as {rec.kind.value}, not an agent"]
        return rec

    status = Status.DISCOVERED
    fams = rec.families
    if len(fams) >= 2 and fams & {Family.RUNTIME, Family.BEHAVIORAL} and not rec.conflicts:
        status = Status.VERIFIED
    else:
        notes.append(f"unverified: evidence families {sorted(f.value for f in fams)}")

    if status == Status.VERIFIED:
        if rec.endpoint and rec.protocol:
            status = Status.REACHABLE
        else:
            notes.append("no invocation contract (endpoint/protocol unknown)")

    if status == Status.REACHABLE:
        blockers = _policy_blockers(rec, policy)
        if blockers:
            notes.extend(blockers)
        else:
            status = Status.AUTHORIZED
    else:
        notes.extend(_policy_blockers(rec, policy))

    rec.status, rec.policy_notes = status, notes
    return rec


def _policy_blockers(rec: AgentRecord, policy: Policy) -> list[str]:
    out = []
    if policy.require_owner and not rec.owner_team:
        out.append("policy: no accountable owner")
    if policy.require_clearance and not rec.data_clearance:
        out.append("policy: data clearance unknown")
    if risky := sorted(set(rec.tools) & policy.blocked_tools):
        out.append(f"policy: blocked tools {risky}")
    runs = rec.total_runs()
    if runs >= policy.min_runs_for_violation_check:
        rate = sum(s.guardrail_violations for s in rec.telemetry.values()) / runs
        if rate > policy.max_guardrail_violation_rate:
            out.append(f"policy: guardrail violation rate {rate:.3f}")
    return out
