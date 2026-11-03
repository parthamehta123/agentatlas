"""agentatlas CLI.

    agentatlas generate  --out data --n 10000
    agentatlas discover  --data data [--no-a2a]
    agentatlas clusters  --data data
    agentatlas select    --data data task "Screen a supplier against sanctions lists" \
                         --require sanctions_screening --classification restricted
    agentatlas select    --data data representative
    agentatlas select    --data data security
    agentatlas benchmark --data data --out reports
    agentatlas serve     --data data
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from agentatlas.selection.engine import TaskSpec


def _atlas(data: Path, include_a2a: bool = True):
    from agentatlas.service import Atlas

    atlas = Atlas()
    atlas.discover(data / "infra", include_a2a)
    return atlas


def _print_rows(rows: list[dict], cols: list[str]) -> None:
    print("  ".join(f"{c:>12}" if c != "name" else f"{c:<28}" for c in cols))
    for r in rows:
        print("  ".join(f"{r.get(c, '')!s:>12}" if c != "name" else f"{r[c][:28]:<28}"
                        for c in cols))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agentatlas")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="synthesise an enterprise agent estate")
    g.add_argument("--out", type=Path, default=Path("data"))
    g.add_argument("--n", type=int, default=10_000)
    g.add_argument("--seed", type=int, default=42)

    d = sub.add_parser("discover", help="run discovery adapters over infra artefacts")
    d.add_argument("--data", type=Path, default=Path("data"))
    d.add_argument("--no-a2a", action="store_true")

    c = sub.add_parser("clusters", help="cluster the registry and summarise")
    c.add_argument("--data", type=Path, default=Path("data"))
    c.add_argument("--k", type=int, default=12)

    s = sub.add_parser("select", help="select top-k agents")
    s.add_argument("--data", type=Path, default=Path("data"))
    s.add_argument("--k", type=int, default=10)
    s.add_argument("mode", choices=["task", "representative", "security"])
    s.add_argument("description", nargs="?", default="")
    s.add_argument("--require", nargs="*", default=[])
    s.add_argument("--category")
    s.add_argument("--classification", default="internal")
    s.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="evaluate every stage against ground truth")
    b.add_argument("--data", type=Path, default=Path("data"))
    b.add_argument("--out", type=Path, default=Path("reports"))

    v = sub.add_parser("serve", help="run the FastAPI service")
    v.add_argument("--data", type=Path, default=Path("data"))
    v.add_argument("--port", type=int, default=8080)

    a = p.parse_args(argv)

    if a.cmd == "generate":
        from agentatlas.synthetic.generate import GenConfig, write_dataset

        print(json.dumps(write_dataset(a.out, GenConfig(n_agents=a.n, seed=a.seed)), indent=1))
    elif a.cmd == "discover":
        rep = _atlas(a.data, not a.no_a2a).last_report
        out = asdict(rep)
        out.pop("records")
        out["conflicts"] = len(out["conflicts"])
        print(json.dumps(out, indent=1))
    elif a.cmd == "clusters":
        atlas = _atlas(a.data)
        atlas.build(a.k)
        for row in atlas.cluster_summary():
            print(f"[{row['cluster']:>2}] {row['size']:>5} agents  "
                  f"auth={row['authorized_share']:.2f}  {', '.join(row['top_capabilities'])}"
                  f"  -> {row['representative']['name']}")
    elif a.cmd == "select":
        atlas = _atlas(a.data)
        if a.mode == "task":
            if not a.description:
                p.error("task mode needs a description")
            from agentatlas.selection.engine import TaskConfig

            res = atlas.select_task(TaskSpec(description=a.description, category=a.category,
                                             required_capabilities=a.require,
                                             data_classification=a.classification),
                                    TaskConfig(k=a.k))
            cols = ["rank", "agent_id", "name", "score", "quality_lb", "runs", "cluster"]
        elif a.mode == "representative":
            res, cols = atlas.select_representative(a.k), ["rank", "agent_id", "name", "why"]
        else:
            res = atlas.select_security(a.k)
            cols = ["rank", "agent_id", "name", "priority", "exposure", "status"]
        if a.json:
            out = asdict(res)
            out["meta"].pop("retrieved", None)
            print(json.dumps(out, indent=1, default=str))
            return
        for f in res.funnel:
            print(f"  {f['stage']:<16} {f['count']:>6}  {f.get('dropped', '')}")
        _print_rows(res.rows, cols)
    elif a.cmd == "benchmark":
        from agentatlas.evaluation.benchmark import run_benchmark, to_markdown

        res = run_benchmark(a.data)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "benchmark.json").write_text(json.dumps(res, indent=1, default=str))
        (a.out / "BENCHMARK.md").write_text(to_markdown(res))
        print(to_markdown(res))
    elif a.cmd == "serve":
        import uvicorn

        os.environ.setdefault("AGENTATLAS_INFRA", str(a.data / "infra"))
        uvicorn.run("agentatlas.api.main:app", host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
