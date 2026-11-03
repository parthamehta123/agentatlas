import json

import numpy as np

from agentatlas.registry.models import Status
from agentatlas.selection.engine import TaskConfig, TaskSpec, mmr
from agentatlas.synthetic.archetypes import CLASSIFICATION_ORDER


def _tasks(data_dir):
    return json.loads((data_dir / "truth/tasks.json").read_text())


def test_task_selection_enforces_hard_constraints(atlas, data_dir):
    for t in _tasks(data_dir):
        spec = TaskSpec(description=t["description"], category=t["category"],
                        required_capabilities=t["required_capabilities"],
                        data_classification=t["data_classification"])
        res = atlas.select_task(spec)
        need = CLASSIFICATION_ORDER.index(t["data_classification"])
        for aid in res.agent_ids:
            r = atlas.registry.get(aid)
            assert r.status == Status.AUTHORIZED
            assert CLASSIFICATION_ORDER.index(r.data_clearance) >= need
            assert set(t["required_capabilities"]) <= set(
                atlas.features.records[atlas.features.index[aid]].all_capabilities())
        assert len(set(res.agent_ids)) == len(res.agent_ids)


def test_needle_agent_is_preserved(atlas, data_dir):
    truth = {a["agent_id"]: a for a in json.loads((data_dir / "truth/agents.json").read_text())}
    needle = next(t for t in _tasks(data_dir) if t["category"] == "sanctions_screening")
    res = atlas.select_task(TaskSpec(description=needle["description"],
                                     category=needle["category"],
                                     required_capabilities=needle["required_capabilities"],
                                     data_classification="restricted"))
    authorized_needles = [a for a, t in truth.items() if t["is_needle"]
                          and (r := atlas.registry.get(a)) and r.status == Status.AUTHORIZED]
    assert set(authorized_needles) <= set(res.agent_ids)


def test_funnel_is_monotone(atlas):
    res = atlas.select_task(TaskSpec(description="answer warehouse sql questions",
                                     required_capabilities=["sql"]), TaskConfig(k=5))
    counts = [f["count"] for f in res.funnel if f["stage"] != "scanned"]
    assert counts[0] >= counts[1] and counts[-1] == len(res.agent_ids) <= 5


def test_mmr_trades_relevance_for_diversity():
    vecs = np.array([[1, 0], [0.99, 0.01], [0, 1]], float)
    scores = np.array([1.0, 0.99, 0.6])
    assert mmr(scores, vecs, 2, lam=1.0) == [0, 1]
    assert mmr(scores, vecs, 2, lam=0.5) == [0, 2]


def test_representative_and_security_modes(atlas):
    rep = atlas.select_representative(10, outlier_slots=1)
    assert len(rep.agent_ids) == 10 and "outlier" in rep.rows[-1]["why"]
    sec = atlas.select_security(10)
    assert len(sec.agent_ids) == 10
    # security testing must be able to target agents that are not yet trusted
    assert any(r["status"] != "AUTHORIZED" for r in sec.rows)
