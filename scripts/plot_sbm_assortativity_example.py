#!/usr/bin/env python3
"""Draw three simple SBMs with decreasing assortativity."""

from __future__ import annotations

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


OUTPUT = REPO_DIR / "results" / "plots" / "current_sweeps" / "sbm_assortativity_example.png"
BLOCK_SIZES = [18, 18, 18]
COLORS = ["#4c78a8", "#f58518", "#54a24b"]


def block_probabilities(p_in: float, p_out: float) -> list[list[float]]:
    return [
        [p_in if i == j else p_out for j in range(len(BLOCK_SIZES))]
        for i in range(len(BLOCK_SIZES))
    ]


def block_positions() -> dict[int, tuple[float, float]]:
    centers = [(-1.25, 0.35), (1.25, 0.35), (0.0, -1.1)]
    positions: dict[int, tuple[float, float]] = {}
    start = 0
    for block_idx, size in enumerate(BLOCK_SIZES):
        clique = nx.circular_layout(nx.cycle_graph(size), scale=0.43, center=centers[block_idx])
        for local_idx in range(size):
            positions[start + local_idx] = tuple(clique[local_idx])
        start += size
    return positions


def node_colors() -> list[str]:
    colors = []
    for block_idx, size in enumerate(BLOCK_SIZES):
        colors.extend([COLORS[block_idx]] * size)
    return colors


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pos = block_positions()
    colors = node_colors()
    cases = [
        ("Strong assortativity", 0.42, 0.015),
        ("Medium assortativity", 0.30, 0.065),
        ("Weak assortativity", 0.20, 0.14),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), squeeze=False)
    for ax, (title, p_in, p_out) in zip(axes[0], cases):
        graph = nx.stochastic_block_model(
            BLOCK_SIZES,
            block_probabilities(p_in, p_out),
            seed=7,
        )
        within_edges = []
        across_edges = []
        for u, v in graph.edges():
            if u // BLOCK_SIZES[0] == v // BLOCK_SIZES[0]:
                within_edges.append((u, v))
            else:
                across_edges.append((u, v))

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=within_edges,
            ax=ax,
            edge_color="#7f8c8d",
            width=0.8,
            alpha=0.34,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=across_edges,
            ax=ax,
            edge_color="#2f2f2f",
            width=0.9,
            alpha=0.50,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=colors,
            node_size=64,
            linewidths=0.55,
            edgecolors="white",
        )
        ax.set_title(f"{title}\n$p_{{in}}={p_in:.2f}$, $p_{{out}}={p_out:.3f}$", fontsize=13)
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(-1.75, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("Stochastic Block Model Assortativity", fontsize=16)
    fig.text(
        0.5,
        0.035,
        "Node colors are planted communities. As p_out approaches p_in, cross-community edges become common and the block structure is harder to recover.",
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
