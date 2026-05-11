#!/usr/bin/env python3
"""Draw simple LFR examples with increasing mixing parameter mu."""

from __future__ import annotations

import math
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = REPO_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


OUTPUT = REPO_DIR / "results" / "plots" / "current_sweeps" / "lfr_mixing_example.png"
PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#9d755d",
    "#bab0ac",
]


def generate_lfr(mu: float, seed: int) -> nx.Graph:
    for offset in range(30):
        try:
            graph = nx.LFR_benchmark_graph(
                n=90,
                tau1=2.5,
                tau2=1.5,
                mu=mu,
                average_degree=8,
                max_degree=25,
                min_community=10,
                max_community=24,
                seed=seed + offset,
            )
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
        except nx.ExceededMaxIterations:
            continue
    raise RuntimeError(f"could not generate LFR graph for mu={mu}")


def community_labels(graph: nx.Graph) -> tuple[dict[int, int], list[list[int]]]:
    communities_by_members = {}
    for node, data in graph.nodes(data=True):
        members = tuple(sorted(data["community"]))
        communities_by_members.setdefault(members, []).append(node)
    communities = sorted(communities_by_members.values(), key=lambda nodes: (-len(nodes), min(nodes)))
    labels = {}
    for idx, nodes in enumerate(communities):
        for node in nodes:
            labels[node] = idx
    return labels, communities


def community_layout(communities: list[list[int]]) -> dict[int, tuple[float, float]]:
    count = len(communities)
    radius = 1.15 if count <= 5 else 1.35
    centers = []
    for idx in range(count):
        angle = 2.0 * math.pi * idx / count + math.pi / 2.0
        centers.append((radius * math.cos(angle), radius * math.sin(angle)))

    positions = {}
    for idx, nodes in enumerate(communities):
        local = nx.circular_layout(nx.cycle_graph(len(nodes)), scale=0.24, center=centers[idx])
        for local_idx, node in enumerate(sorted(nodes)):
            positions[node] = tuple(local[local_idx])
    return positions


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        ("Low mixing", 0.08),
        ("Medium mixing", 0.25),
        ("High mixing", 0.45),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), squeeze=False)
    for ax, (title, mu) in zip(axes[0], cases):
        graph = generate_lfr(mu, seed=11)
        labels, communities = community_labels(graph)
        positions = community_layout(communities)
        node_colors = [PALETTE[labels[node] % len(PALETTE)] for node in graph.nodes()]
        internal_edges = []
        external_edges = []
        for u, v in graph.edges():
            if labels[u] == labels[v]:
                internal_edges.append((u, v))
            else:
                external_edges.append((u, v))

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=internal_edges,
            ax=ax,
            edge_color="#7f8c8d",
            width=0.55,
            alpha=0.30,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=external_edges,
            ax=ax,
            edge_color="#2f2f2f",
            width=0.75,
            alpha=0.52,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=node_colors,
            node_size=42,
            linewidths=0.45,
            edgecolors="white",
        )
        ax.set_title(
            f"{title}\n$\\mu={mu:.2f}$, external edges={len(external_edges)}",
            fontsize=13,
        )
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("LFR Mixing Parameter", fontsize=16)
    fig.text(
        0.5,
        0.035,
        "Node colors are planted communities. Larger mu means more edges leave each node's community, so boundaries become noisier.",
        ha="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0.075, 1, 0.90])
    fig.savefig(OUTPUT, dpi=240)
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
