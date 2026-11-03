"""Discovery adapter contract.

Adapters only *observe*. They emit :class:`Candidate` records with identity hints and
evidence families; they never decide whether something is an agent or trusted. That
is the registry's job (identity resolution + verification).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

from agentatlas.registry.models import Candidate

_SUFFIX = re.compile(r"(-(svc|service|agent|v\d+))+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Normalise names across systems: 'Fraud Scout 00042', 'FRAUD_SCOUT_00042',
    'fraud-scout-00042-svc' all become 'fraud-scout-00042'."""
    s = _NON_ALNUM.sub("-", name.lower()).strip("-")
    return _SUFFIX.sub("", s)


class DiscoveryAdapter(ABC):
    name: str

    @abstractmethod
    def collect(self) -> Iterator[Candidate]: ...
