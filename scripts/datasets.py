#!/usr/bin/env python3
"""Generate and import graph datasets into a common bundle format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.graph import Graph  # noqa: E402
from data.io import save_graph_bundle  # noqa: E402


def _load_networkx():
    try:
        import networkx as nx  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "networkx is required for this subcommand but is not installed. "
            "Install project dependencies first."
        ) from exc
    return nx


def _save_bundle(
    output: Path,
    *,
    adjacency: sp.csr_matrix,
    target: np.ndarray,
    num_clusters: int,
    name: str,
    metadata: dict[str, Any],
) -> Path:
    graph = Graph(
        adjacency=adjacency,
        num_nodes=adjacency.shape[0],
        name=name,
        metadata=metadata,
    )
    return save_graph_bundle(
        output,
        graph=graph,
        target=target,
        num_clusters=num_clusters,
        metadata=metadata,
    )


def _normalize_csr(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    adjacency = adjacency.tocsr().astype(float)
    adjacency = ((adjacency + adjacency.T) > 0).astype(float).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def generate_sbm(
    *,
    sizes: list[int],
    p_in: float,
    p_out: float,
    seed: int,
    output: Path,
    name: str | None,
) -> Path:
    rng = np.random.default_rng(seed)
    labels = np.concatenate(
        [np.full(size, cluster_id, dtype=int) for cluster_id, size in enumerate(sizes)]
    )

    blocks: list[list[sp.csr_matrix]] = []
    for i, size_i in enumerate(sizes):
        row_blocks: list[sp.csr_matrix] = []
        for j, size_j in enumerate(sizes):
            if i == j:
                upper = rng.random((size_i, size_i))
                mat = np.triu((upper < p_in).astype(float), k=1)
                block = sp.csr_matrix(mat + mat.T)
            elif i < j:
                mat = (rng.random((size_i, size_j)) < p_out).astype(float)
                block = sp.csr_matrix(mat)
            else:
                block = blocks[j][i].T.tocsr()
            row_blocks.append(block)
        blocks.append(row_blocks)

    adjacency = sp.bmat(blocks, format="csr")
    adjacency = _normalize_csr(adjacency)
    metadata = {
        "kind": "synthetic_sbm",
        "sizes": sizes,
        "p_in": p_in,
        "p_out": p_out,
        "seed": seed,
    }
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=labels,
        num_clusters=len(sizes),
        name=name or output.stem,
        metadata=metadata,
    )


def _communities_to_labels(communities: list[frozenset[int]]) -> np.ndarray:
    node_ids = sorted(set().union(*communities))
    labels = np.full(len(node_ids), -1, dtype=int)
    for cluster_id, community in enumerate(communities):
        for node in community:
            labels[int(node)] = cluster_id
    if np.any(labels < 0):
        raise ValueError("failed to assign every node to an LFR community")
    return labels


def generate_lfr(
    *,
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    seed: int,
    output: Path,
    average_degree: int | None,
    min_degree: int | None,
    min_community: int,
    max_iters: int,
    name: str | None,
) -> Path:
    nx = _load_networkx()

    lfr_kwargs: dict[str, Any] = {
        "n": n,
        "tau1": tau1,
        "tau2": tau2,
        "mu": mu,
        "min_community": min_community,
        "max_iters": max_iters,
        "seed": seed,
    }
    if average_degree is not None:
        lfr_kwargs["average_degree"] = average_degree
    if min_degree is not None:
        lfr_kwargs["min_degree"] = min_degree

    graph_nx = nx.generators.community.LFR_benchmark_graph(**lfr_kwargs)
    communities = {
        frozenset(graph_nx.nodes[node]["community"]) for node in graph_nx.nodes()
    }
    ordered_communities = sorted(communities, key=lambda community: (len(community), min(community)))
    labels = _communities_to_labels(ordered_communities)
    adjacency = nx.to_scipy_sparse_array(
        graph_nx,
        nodelist=sorted(graph_nx.nodes()),
        format="csr",
        dtype=float,
    )
    adjacency = _normalize_csr(sp.csr_matrix(adjacency))
    metadata = {
        "kind": "synthetic_lfr",
        "n": n,
        "tau1": tau1,
        "tau2": tau2,
        "mu": mu,
        "seed": seed,
        "average_degree": average_degree,
        "min_degree": min_degree,
        "min_community": min_community,
        "max_iters": max_iters,
    }
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=labels,
        num_clusters=len(ordered_communities),
        name=name or output.stem,
        metadata=metadata,
    )


def generate_ring_of_cliques(
    *,
    num_cliques: int,
    clique_size: int,
    output: Path,
    name: str | None,
) -> Path:
    if num_cliques < 2 or clique_size < 2:
        raise ValueError("ring_of_cliques requires num_cliques >= 2 and clique_size >= 2")

    num_nodes = num_cliques * clique_size
    rows: list[int] = []
    cols: list[int] = []
    labels = np.empty(num_nodes, dtype=int)

    for clique in range(num_cliques):
        start = clique * clique_size
        nodes = np.arange(start, start + clique_size, dtype=int)
        labels[nodes] = clique

        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                u = int(nodes[i])
                v = int(nodes[j])
                rows.extend((u, v))
                cols.extend((v, u))

        bridge_u = int(nodes[-1])
        next_start = ((clique + 1) % num_cliques) * clique_size
        bridge_v = int(next_start)
        rows.extend((bridge_u, bridge_v))
        cols.extend((bridge_v, bridge_u))

    data = np.ones(len(rows), dtype=float)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency = _normalize_csr(adjacency)
    metadata = {
        "kind": "ring_of_cliques",
        "num_cliques": num_cliques,
        "clique_size": clique_size,
    }
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=labels,
        num_clusters=num_cliques,
        name=name or output.stem,
        metadata=metadata,
    )


def generate_core_periphery_cliques(
    *,
    core_size: int,
    num_periphery: int,
    periphery_size: int,
    output: Path,
    name: str | None,
) -> Path:
    if core_size < 2 or num_periphery < 1 or periphery_size < 2:
        raise ValueError(
            "core_periphery_cliques requires core_size >= 2, num_periphery >= 1, and periphery_size >= 2"
        )

    num_nodes = core_size + num_periphery * periphery_size
    rows: list[int] = []
    cols: list[int] = []
    labels = np.empty(num_nodes, dtype=int)

    core_nodes = np.arange(core_size, dtype=int)
    labels[core_nodes] = 0
    for i in range(core_size):
        for j in range(i + 1, core_size):
            u = int(core_nodes[i])
            v = int(core_nodes[j])
            rows.extend((u, v))
            cols.extend((v, u))

    for block in range(num_periphery):
        start = core_size + block * periphery_size
        nodes = np.arange(start, start + periphery_size, dtype=int)
        labels[nodes] = block + 1

        for i in range(periphery_size):
            for j in range(i + 1, periphery_size):
                u = int(nodes[i])
                v = int(nodes[j])
                rows.extend((u, v))
                cols.extend((v, u))

        anchor = int(core_nodes[block % core_size])
        attachment = int(nodes[0])
        rows.extend((anchor, attachment))
        cols.extend((attachment, anchor))

    data = np.ones(len(rows), dtype=float)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency = _normalize_csr(adjacency)
    metadata = {
        "kind": "core_periphery_cliques",
        "core_size": core_size,
        "num_periphery": num_periphery,
        "periphery_size": periphery_size,
    }
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=labels,
        num_clusters=num_periphery + 1,
        name=name or output.stem,
        metadata=metadata,
    )


def import_builtin_social_graph(
    *,
    source: str,
    output: Path,
    name: str | None,
) -> Path:
    nx = _load_networkx()

    if source == "karate_club":
        graph_nx = nx.karate_club_graph()
        nodes = sorted(graph_nx.nodes())
        club_to_id = {"Mr. Hi": 0, "Officer": 1}
        labels = np.array([club_to_id[graph_nx.nodes[node]["club"]] for node in nodes])
        num_clusters = 2
        metadata = {
            "kind": "builtin_social",
            "source": source,
            "label_source": "club",
        }
    else:
        raise ValueError(f"unsupported builtin social graph source: {source!r}")

    adjacency = nx.to_scipy_sparse_array(
        graph_nx,
        nodelist=nodes,
        format="csr",
        dtype=float,
    )
    adjacency = _normalize_csr(sp.csr_matrix(adjacency))
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=labels,
        num_clusters=num_clusters,
        name=name or output.stem,
        metadata=metadata,
    )


def _read_labels_file(path: Path, node_to_idx: dict[str, int]) -> tuple[np.ndarray, int]:
    labels = np.full(len(node_to_idx), -1, dtype=int)
    label_to_id: dict[str, int] = {}

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                node_id, label_name = [part.strip() for part in line.split(",", maxsplit=1)]
            else:
                parts = [part.strip() for part in line.split() if part.strip()]
                if len(parts) < 2:
                    raise ValueError(
                        "labels file rows must be '<node> <label>' or '<node>,<label>'; "
                        f"got {line!r}"
                    )
                node_id = parts[0]
                label_name = " ".join(parts[1:])
            try:
                idx = node_to_idx[node_id]
            except KeyError as exc:
                raise ValueError(f"label file referenced unknown node {node_id!r}") from exc
            if label_name not in label_to_id:
                label_to_id[label_name] = len(label_to_id)
            labels[idx] = label_to_id[label_name]

    if np.any(labels < 0):
        missing = int(np.sum(labels < 0))
        raise ValueError(f"labels file did not cover every node; missing {missing} nodes")
    return labels, len(label_to_id)


def import_edgelist(
    *,
    input_path: Path,
    output: Path,
    labels_path: Path | None,
    num_clusters: int | None,
    delimiter: str | None,
    comment_prefix: str,
    name: str | None,
) -> Path:
    node_to_idx: dict[str, int] = {}
    edges: set[tuple[int, int]] = set()

    def node_index(node_id: str) -> int:
        if node_id not in node_to_idx:
            node_to_idx[node_id] = len(node_to_idx)
        return node_to_idx[node_id]

    with open(input_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(comment_prefix):
                continue
            if delimiter is None:
                parts = line.split()
            else:
                parts = [part.strip() for part in line.split(delimiter) if part.strip()]
            if len(parts) < 2:
                raise ValueError(f"expected at least two columns per edge; got {line!r}")
            u = node_index(parts[0])
            v = node_index(parts[1])
            if u == v:
                continue
            a, b = sorted((u, v))
            edges.add((a, b))

    num_nodes = len(node_to_idx)
    rows = np.array([u for u, v in edges] + [v for u, v in edges], dtype=int)
    cols = np.array([v for u, v in edges] + [u for u, v in edges], dtype=int)
    data = np.ones(len(rows), dtype=float)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adjacency = _normalize_csr(adjacency)

    if labels_path is not None:
        target, inferred_num_clusters = _read_labels_file(labels_path, node_to_idx)
    else:
        target = np.full(num_nodes, -1, dtype=int)
        inferred_num_clusters = int(num_clusters or 1)

    final_num_clusters = int(num_clusters or inferred_num_clusters)
    metadata = {
        "kind": "imported_edgelist",
        "input_path": str(input_path),
        "labels_path": str(labels_path) if labels_path else None,
        "original_node_ids": json.dumps(
            {idx: node_id for node_id, idx in node_to_idx.items()},
            sort_keys=True,
        ),
    }
    return _save_bundle(
        output,
        adjacency=adjacency,
        target=target,
        num_clusters=final_num_clusters,
        name=name or output.stem,
        metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or import graph datasets into a common .npz bundle format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sbm = subparsers.add_parser("sbm", help="generate a stochastic block model graph")
    sbm.add_argument("--sizes", required=True, help="comma-separated block sizes, e.g. 40,40,40")
    sbm.add_argument("--p-in", type=float, required=True, help="intra-block edge probability")
    sbm.add_argument("--p-out", type=float, required=True, help="inter-block edge probability")
    sbm.add_argument("--seed", type=int, default=0)
    sbm.add_argument("--output", type=Path, required=True)
    sbm.add_argument("--name")

    lfr = subparsers.add_parser("lfr", help="generate an LFR benchmark graph")
    lfr.add_argument("--n", type=int, required=True)
    lfr.add_argument("--tau1", type=float, default=3.0)
    lfr.add_argument("--tau2", type=float, default=1.5)
    lfr.add_argument("--mu", type=float, required=True)
    lfr.add_argument("--seed", type=int, default=0)
    lfr.add_argument("--average-degree", type=int)
    lfr.add_argument("--min-degree", type=int)
    lfr.add_argument("--min-community", type=int, default=10)
    lfr.add_argument("--max-iters", type=int, default=500)
    lfr.add_argument("--output", type=Path, required=True)
    lfr.add_argument("--name")

    ring = subparsers.add_parser(
        "ring_of_cliques",
        help="generate a ring of weakly bridged cliques",
    )
    ring.add_argument("--num-cliques", type=int, required=True)
    ring.add_argument("--clique-size", type=int, required=True)
    ring.add_argument("--output", type=Path, required=True)
    ring.add_argument("--name")

    corep = subparsers.add_parser(
        "core_periphery_cliques",
        help="generate a dense core with attached clique peripherals",
    )
    corep.add_argument("--core-size", type=int, required=True)
    corep.add_argument("--num-periphery", type=int, required=True)
    corep.add_argument("--periphery-size", type=int, required=True)
    corep.add_argument("--output", type=Path, required=True)
    corep.add_argument("--name")

    social = subparsers.add_parser(
        "social",
        help="import a builtin social graph with labels when available",
    )
    social.add_argument("--source", choices=["karate_club"], required=True)
    social.add_argument("--output", type=Path, required=True)
    social.add_argument("--name")

    edgelist = subparsers.add_parser(
        "edgelist",
        help="parse a local edge list and optional labels file into a graph bundle",
    )
    edgelist.add_argument("--input", type=Path, required=True)
    edgelist.add_argument("--output", type=Path, required=True)
    edgelist.add_argument("--labels", type=Path)
    edgelist.add_argument("--num-clusters", type=int)
    edgelist.add_argument("--delimiter")
    edgelist.add_argument("--comment-prefix", default="#")
    edgelist.add_argument("--name")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sbm":
        sizes = [int(part.strip()) for part in args.sizes.split(",") if part.strip()]
        output = generate_sbm(
            sizes=sizes,
            p_in=args.p_in,
            p_out=args.p_out,
            seed=args.seed,
            output=args.output,
            name=args.name,
        )
    elif args.command == "lfr":
        output = generate_lfr(
            n=args.n,
            tau1=args.tau1,
            tau2=args.tau2,
            mu=args.mu,
            seed=args.seed,
            output=args.output,
            average_degree=args.average_degree,
            min_degree=args.min_degree,
            min_community=args.min_community,
            max_iters=args.max_iters,
            name=args.name,
        )
    elif args.command == "social":
        output = import_builtin_social_graph(
            source=args.source,
            output=args.output,
            name=args.name,
        )
    elif args.command == "ring_of_cliques":
        output = generate_ring_of_cliques(
            num_cliques=args.num_cliques,
            clique_size=args.clique_size,
            output=args.output,
            name=args.name,
        )
    elif args.command == "core_periphery_cliques":
        output = generate_core_periphery_cliques(
            core_size=args.core_size,
            num_periphery=args.num_periphery,
            periphery_size=args.periphery_size,
            output=args.output,
            name=args.name,
        )
    elif args.command == "edgelist":
        output = import_edgelist(
            input_path=args.input,
            output=args.output,
            labels_path=args.labels,
            num_clusters=args.num_clusters,
            delimiter=args.delimiter,
            comment_prefix=args.comment_prefix,
            name=args.name,
        )
    else:
        raise AssertionError(f"unhandled command {args.command!r}")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
