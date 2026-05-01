"""Name-keyed registries for plug-and-play datasets, algorithms, and evals."""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

DATASETS: dict[str, type] = {}
ALGORITHMS: dict[str, type] = {}
BASELINES: dict[str, type] = {}
EVALS: dict[str, type] = {}


def _register(registry: dict[str, type], name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in registry:
            raise ValueError(
                f"name {name!r} already registered as {registry[name].__name__}"
            )
        registry[name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    return _register(DATASETS, name)


def register_algorithm(name: str) -> Callable[[type[T]], type[T]]:
    return _register(ALGORITHMS, name)


def register_baseline(name: str) -> Callable[[type[T]], type[T]]:
    return _register(BASELINES, name)


def register_eval(name: str) -> Callable[[type[T]], type[T]]:
    return _register(EVALS, name)


def resolve(registry: dict[str, type], key: str) -> type:
    try:
        return registry[key]
    except KeyError:
        available = ", ".join(sorted(registry)) or "(none registered)"
        raise KeyError(
            f"unknown registry key {key!r}; available: {available}"
        ) from None
