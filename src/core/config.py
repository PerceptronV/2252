"""YAML config schema and loader for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ComponentSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    dataset: ComponentSpec
    algorithm: ComponentSpec
    evals: list[ComponentSpec]
    raw: dict[str, Any]  # original YAML for round-tripping into results

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        for required in ("name", "seed", "dataset", "algorithm", "evals"):
            if required not in raw:
                raise ValueError(f"config missing required field: {required!r}")

        def _component(d: dict[str, Any]) -> ComponentSpec:
            if "type" not in d:
                raise ValueError(f"component spec missing 'type': {d!r}")
            return ComponentSpec(type=d["type"], params=dict(d.get("params") or {}))

        evals_raw = raw["evals"]
        if not isinstance(evals_raw, list) or not evals_raw:
            raise ValueError("'evals' must be a non-empty list")

        return cls(
            name=str(raw["name"]),
            seed=int(raw["seed"]),
            dataset=_component(raw["dataset"]),
            algorithm=_component(raw["algorithm"]),
            evals=[_component(e) for e in evals_raw],
            raw=raw,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config at {path} did not parse to a mapping")
    return ExperimentConfig.from_dict(raw)
