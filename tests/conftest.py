from pathlib import Path

import pytest

from agentatlas.synthetic.generate import GenConfig, write_dataset

SMALL = GenConfig(n_agents=1500, n_plain_services=60, n_mcp_servers=20, n_batch_jobs=20,
                  lucky_count=15, seed=7)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("estate")
    write_dataset(out, SMALL)
    return out


@pytest.fixture(scope="session")
def atlas(data_dir):
    from agentatlas.service import Atlas

    a = Atlas()
    a.discover(data_dir / "infra")
    a.build(12)
    return a
