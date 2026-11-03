import json

import pytest

from agentatlas.discovery.adapters import KubernetesAdapter, adapters_from_dir
from agentatlas.discovery.base import slugify
from agentatlas.discovery.pipeline import run_discovery
from agentatlas.evaluation.benchmark import Truth, discovery_metrics
from agentatlas.registry.identity import resolve
from agentatlas.registry.models import Candidate, Family, Kind, Status
from agentatlas.registry.verification import Policy, assess


@pytest.mark.parametrize("name", ["Fraud Scout 00042", "FRAUD_SCOUT_00042",
                                  "fraud-scout-00042-svc", "fraud-scout-00042-v2"])
def test_slugify_normalises_name_variants(name):
    assert slugify(name) == "fraud-scout-00042"


def _cand(source, keys, kind=Kind.AGENT, fams=(Family.DECLARED,), replicas=0, **attrs):
    return Candidate(source=source, source_ref=f"{source}-{min(keys)}", kind=kind,
                     families=set(fams), identity_keys=set(keys), attributes=attrs,
                     replicas=replicas)


def test_replicas_and_clusters_collapse_into_one_logical_agent():
    cands = [
        _cand("registration", {"id:a1", "slug:fraud-scout-1"}, name="Fraud Scout 1"),
        _cand("kubernetes", {"slug:fraud-scout-1"}, fams=(Family.RUNTIME,), replicas=3),
        _cand("kubernetes", {"slug:fraud-scout-1"}, fams=(Family.RUNTIME,), replicas=2),
    ]
    recs, conflicts = resolve(cands)
    assert len(recs) == 1 and not conflicts
    assert recs[0].agent_id == "a1" and recs[0].instances == 5


def test_two_explicit_ids_are_never_merged():
    cands = [_cand("registration", {"id:a1", "slug:shared-name"}),
             _cand("registration", {"id:a2", "slug:shared-name"})]
    recs, conflicts = resolve(cands)
    assert {r.agent_id for r in recs} == {"a1", "a2"}
    assert conflicts


def test_kubernetes_multi_agent_host_and_mcp_server(tmp_path):
    def item(name, labels=None, ann=None, env=("LITELLM_BASE_URL",)):
        return {"metadata": {"name": name, "namespace": "finance", "labels": labels or {},
                             "annotations": ann or {}},
                "spec": {"replicas": 4, "template": {"spec": {"containers": [
                    {"image": "x", "env": [{"name": e} for e in env]}]}}}}

    path = tmp_path / "k8s.test.json"
    path.write_text(json.dumps({"items": [
        item("host", ann={"agentatlas.io/agents": "agent-a,agent-b,agent-c"}),
        item("tools", labels={"agentatlas.io/kind": "mcp-server"}),
        item("web", env=("DATABASE_URL",)),
    ]}))
    kinds = [c.kind for c in KubernetesAdapter([path]).collect()]
    assert kinds.count(Kind.AGENT) == 3
    assert Kind.MCP_SERVER in kinds and Kind.SERVICE in kinds


def test_status_ladder_and_policy_block():
    rec, _ = resolve([
        _cand("registration", {"id:a1", "slug:x"}, owner_team="t", data_clearance="internal",
              endpoint="https://x", protocol="http", tools=["kb_search"]),
        _cand("kubernetes", {"slug:x"}, fams=(Family.RUNTIME,)),
    ])
    assert assess(rec[0], Policy()).status == Status.AUTHORIZED
    rec[0].tools.append("shell_exec")
    assert assess(rec[0], Policy()).status == Status.REACHABLE
    alone, _ = resolve([_cand("cmdb", {"slug:y"}, fams=(Family.OWNERSHIP,))])
    assert assess(alone[0], Policy()).status == Status.DISCOVERED


def test_end_to_end_discovery_without_a2a(data_dir):
    truth = Truth.load(data_dir)
    full = discovery_metrics(run_discovery(adapters_from_dir(data_dir / "infra")).records,
                             truth)
    no_a2a = discovery_metrics(run_discovery(
        adapters_from_dir(data_dir / "infra", include_a2a=False)).records, truth)
    assert full["precision"] == 1.0 and full["duplicates"] == 0
    assert no_a2a["recall"] > 0.97  # discovery does not need A2A ...
    assert no_a2a["reachable_share"] <= full["reachable_share"]  # ... invocation benefits
    assert full["shadow_agents_found"] > 0
