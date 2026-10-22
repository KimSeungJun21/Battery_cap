#!/usr/bin/env python3
"""Render the cached Battery IMSI route graph as a simple NetworkX PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    default_cache_dir = workspace / "zivid_capture" / "object_occupancy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir,
        help="Directory containing the IMSI joint and route caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            workspace
            / "rb_ws"
            / "docs"
            / "Battery_imsi_networkx_graph.png"
        ),
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help="Lowest-cost outgoing edges to display per candidate.",
    )
    return parser.parse_args()


def load_graph_data(cache_dir: Path):
    route_npz = cache_dir / "nbv_precomputed_route_graph.npz"
    route_json = cache_dir / "nbv_precomputed_route_graph.json"
    joint_json = cache_dir / "nbv_precomputed_joint_views.json"

    with np.load(route_npz, allow_pickle=False) as payload:
        names = [str(value) for value in payload["candidate_names"]]
        edge_costs = np.asarray(payload["edge_cost_matrix"], dtype=float)
        initial_costs = np.asarray(payload["initial_edge_costs"], dtype=float)

    with route_json.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    route = list(summary["initial_plan"]["route"])

    with joint_json.open(encoding="utf-8") as stream:
        joint_cache = json.load(stream)
    joints_by_name = {
        attempt["pose_name"]: np.asarray(attempt["positions"], dtype=float)
        for attempt in joint_cache["attempts"]
        if attempt.get("success") and attempt.get("positions")
    }
    joint_matrix = np.stack([joints_by_name[name] for name in names])
    return names, edge_costs, initial_costs, route, joint_matrix, summary


def joint_space_layout(joint_matrix: np.ndarray) -> dict[int, np.ndarray]:
    """Place candidates with a deterministic two-dimensional joint-space PCA."""

    centered = joint_matrix - joint_matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-9, scale, 1.0)
    _, _, vectors = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ vectors[:2].T
    for axis in range(2):
        low = float(coordinates[:, axis].min())
        high = float(coordinates[:, axis].max())
        if high - low > 1e-9:
            coordinates[:, axis] = (
                2.0 * (coordinates[:, axis] - low) / (high - low) - 1.0
            )
    coordinates[:, 0] *= 1.25
    coordinates[:, 1] *= 0.90
    return {index: coordinates[index] for index in range(len(coordinates))}


def lowest_cost_edges(costs: np.ndarray, count: int):
    count = max(1, int(count))
    edges = []
    for source in range(costs.shape[0]):
        targets = [
            target
            for target in np.argsort(costs[source])
            if target != source and np.isfinite(costs[source, target])
        ][:count]
        edges.extend((source, target) for target in targets)
    return edges


def render(args: argparse.Namespace) -> None:
    (
        names,
        edge_costs,
        initial_costs,
        route,
        joint_matrix,
        summary,
    ) = load_graph_data(args.cache_dir)
    name_to_node = {name: index for index, name in enumerate(names)}
    route_nodes = [name_to_node[name] for name in route]
    route_edges = list(zip(route_nodes[:-1], route_nodes[1:]))

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(names)))
    displayed_edges = lowest_cost_edges(edge_costs, args.neighbors)
    graph.add_edges_from(displayed_edges)

    positions = joint_space_layout(joint_matrix)
    minimum_x = min(point[0] for point in positions.values())
    start_node = "START"
    positions[start_node] = np.array([minimum_x - 0.55, 0.0])

    cheapest_from_start = [
        int(index)
        for index in np.argsort(initial_costs)
        if np.isfinite(initial_costs[index])
    ][: max(3, args.neighbors)]

    figure, axis = plt.subplots(figsize=(18, 12), dpi=160)
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=displayed_edges,
        ax=axis,
        edge_color="#94a3b8",
        width=0.8,
        alpha=0.30,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=8,
        connectionstyle="arc3,rad=0.045",
        node_size=640,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[(start_node, node) for node in cheapest_from_start],
        ax=axis,
        edge_color="#38bdf8",
        width=1.3,
        alpha=0.42,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        node_size=640,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=list(range(len(names))),
        ax=axis,
        node_color="#dbeafe",
        edgecolors="#2563eb",
        linewidths=1.3,
        node_size=640,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=route_nodes,
        ax=axis,
        node_color="#bbf7d0",
        edgecolors="#059669",
        linewidths=2.4,
        node_size=920,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[start_node],
        ax=axis,
        node_color="#fde68a",
        edgecolors="#d97706",
        linewidths=2.4,
        node_size=1050,
    )

    highlighted_graph = nx.DiGraph()
    highlighted_graph.add_edges_from([(start_node, route_nodes[0]), *route_edges])
    nx.draw_networkx_edges(
        highlighted_graph,
        positions,
        ax=axis,
        edge_color="#059669",
        width=4.0,
        alpha=0.96,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        connectionstyle="arc3,rad=0.02",
        node_size=920,
    )

    node_labels = {node: f"{node + 1:02d}" for node in range(len(names))}
    node_labels[start_node] = "S"
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=node_labels,
        ax=axis,
        font_size=8,
        font_weight="bold",
        font_color="#0f172a",
    )

    annotation_offsets = [(10, 20), (12, -34), (12, 20), (12, -34)]
    for order, (node, offset) in enumerate(
        zip(route_nodes, annotation_offsets),
        start=1,
    ):
        axis.annotate(
            f"R{order}: {names[node]}",
            xy=positions[node],
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#065f46",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "#ecfdf5",
                "edgecolor": "#6ee7b7",
                "alpha": 0.95,
            },
        )

    pair_edge_count = int(np.isfinite(edge_costs).sum())
    route_cost = float(summary["initial_plan"]["cost"])
    axis.set_title(
        "Battery IMSI Candidate Graph",
        fontsize=23,
        fontweight="bold",
        color="#0f172a",
        pad=24,
    )
    axis.text(
        0.5,
        1.01,
        (
            f"{len(names)} nodes · {pair_edge_count:,} directed pair edges · "
            f"initial Beam route: {len(route)} views / cost {route_cost:.3f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )
    axis.text(
        0.01,
        0.01,
        (
            f"For readability: {args.neighbors} lowest-cost outgoing edges "
            "per node are shown. Green arrows = selected Beam route."
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#64748b",
    )
    axis.margins(0.16)
    axis.set_axis_off()
    figure.tight_layout(pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    render(parse_args())
