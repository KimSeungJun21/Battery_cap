#!/usr/bin/env python3
"""Render a readable 15-node subgraph of the Battery IMSI route graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from visualize_battery_imsi_networkx import load_graph_data
from visualize_battery_imsi_networkx_summary import short_pose_label


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "zivid_capture" / "object_occupancy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            workspace
            / "rb_ws"
            / "docs"
            / "Battery_imsi_networkx_graph_15nodes.png"
        ),
    )
    parser.add_argument(
        "--total-nodes",
        type=int,
        default=15,
        help="Total displayed nodes including START.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=2,
        help="Lowest-cost outgoing edges shown per displayed candidate.",
    )
    return parser.parse_args()


def choose_candidate_subset(
    edge_costs: np.ndarray,
    initial_costs: np.ndarray,
    route_indices: list[int],
    candidate_count: int,
) -> list[int]:
    """Keep the route and add motion-near alternatives from the real graph."""

    route_set = set(route_indices)

    def relevance(index: int) -> tuple[float, float, int]:
        route_connections = [
            float(edge_costs[index, route_index])
            for route_index in route_indices
        ] + [
            float(edge_costs[route_index, index])
            for route_index in route_indices
        ]
        return (
            min(float(initial_costs[index]), *route_connections),
            float(initial_costs[index]),
            index,
        )

    alternatives = sorted(
        (index for index in range(len(initial_costs)) if index not in route_set),
        key=relevance,
    )
    keep_alternatives = max(0, candidate_count - len(route_indices))
    return [*route_indices, *alternatives[:keep_alternatives]]


def candidate_edges(
    edge_costs: np.ndarray,
    displayed_candidates: list[int],
    count: int,
) -> list[tuple[int, int]]:
    allowed = set(displayed_candidates)
    edges: set[tuple[int, int]] = set()
    for source in displayed_candidates:
        targets = [
            target
            for target in np.argsort(edge_costs[source])
            if target in allowed
            and target != source
            and np.isfinite(edge_costs[source, target])
        ][: max(1, count)]
        edges.update((source, target) for target in targets)
    return sorted(edges)


def render(args: argparse.Namespace) -> None:
    names, edge_costs, initial_costs, route, _, summary = load_graph_data(
        args.cache_dir
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    route_indices = [name_to_index[name] for name in route]
    candidate_count = max(len(route_indices), int(args.total_nodes) - 1)
    displayed = choose_candidate_subset(
        edge_costs,
        initial_costs,
        route_indices,
        candidate_count,
    )
    alternatives = [index for index in displayed if index not in route_indices]

    start = "START"
    graph = nx.DiGraph()
    graph.add_nodes_from([start, *displayed])

    background_edges = candidate_edges(
        edge_costs,
        displayed,
        args.neighbors,
    )
    route_edges = list(zip(route_indices[:-1], route_indices[1:]))
    route_edge_set = set(route_edges)
    background_edges = [
        edge for edge in background_edges if edge not in route_edge_set
    ]

    start_targets = [
        int(index)
        for index in np.argsort(initial_costs)
        if index in set(displayed) and np.isfinite(initial_costs[index])
    ][:3]
    start_edges = [
        (start, target) for target in start_targets if target != route_indices[0]
    ]

    graph.add_edges_from(background_edges)
    graph.add_edges_from(start_edges)
    graph.add_edges_from([(start, route_indices[0]), *route_edges])

    positions: dict[object, tuple[float, float]] = {
        start: (-3.35, 0.0),
    }
    route_x = np.linspace(-2.0, 2.65, len(route_indices))
    route_y = [0.10, -0.10, 0.12, -0.06][: len(route_indices)]
    positions.update(
        {
            node: (float(x), float(y))
            for node, x, y in zip(route_indices, route_x, route_y)
        }
    )
    alternative_slots = [
        (-2.55, 1.28),
        (-1.35, 1.48),
        (-0.10, 1.32),
        (1.15, 1.48),
        (2.40, 1.25),
        (-2.55, -1.22),
        (-1.35, -1.42),
        (-0.10, -1.30),
        (1.15, -1.43),
        (2.40, -1.22),
    ]
    if len(alternatives) > len(alternative_slots):
        raise ValueError("This presentation layout supports at most 10 alternatives")
    positions.update(
        {
            node: alternative_slots[position]
            for position, node in enumerate(alternatives)
        }
    )

    figure, axis = plt.subplots(figsize=(16, 9.2), dpi=170)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=background_edges,
        ax=axis,
        edge_color="#94a3b8",
        width=1.4,
        alpha=0.58,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=13,
        connectionstyle="arc3,rad=0.075",
        node_size=1200,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=start_edges,
        ax=axis,
        edge_color="#7dd3fc",
        width=1.8,
        alpha=0.78,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        connectionstyle="arc3,rad=0.04",
        node_size=1200,
    )
    selected_edges = [(start, route_indices[0]), *route_edges]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=selected_edges,
        ax=axis,
        edge_color="#059669",
        width=5.2,
        alpha=0.98,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=24,
        connectionstyle="arc3,rad=0.0",
        node_size=2100,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=alternatives,
        ax=axis,
        node_color="#dbeafe",
        edgecolors="#2563eb",
        linewidths=1.8,
        node_size=1250,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=route_indices,
        ax=axis,
        node_color="#dcfce7",
        edgecolors="#059669",
        linewidths=2.8,
        node_size=2100,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[start],
        ax=axis,
        node_color="#fef3c7",
        edgecolors="#d97706",
        linewidths=2.8,
        node_size=2100,
    )

    labels = {start: "START"}
    labels.update(
        {
            node: f"A{order}"
            for order, node in enumerate(alternatives, start=1)
        }
    )
    labels.update(
        {
            node: f"R{order}\n{short_pose_label(name)}"
            for order, (node, name) in enumerate(
                zip(route_indices, route),
                start=1,
            )
        }
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=9.5,
        font_weight="bold",
        font_color="#0f172a",
    )

    route_cost_labels = {}
    for edge_position, edge in enumerate(selected_edges):
        if edge_position == 0:
            cost = float(initial_costs[route_indices[0]])
        else:
            source = route_indices[edge_position - 1]
            target = route_indices[edge_position]
            cost = float(edge_costs[source, target])
        route_cost_labels[edge] = f"{cost:.2f} s"
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=route_cost_labels,
        ax=axis,
        font_size=9.5,
        font_color="#047857",
        font_weight="bold",
        rotate=False,
        label_pos=0.5,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "#ffffff",
            "edgecolor": "#a7f3d0",
            "alpha": 0.96,
        },
    )

    total_displayed = 1 + len(displayed)
    route_cost = float(summary["initial_plan"]["cost"])
    axis.set_title(
        f"Motion-Aware NBV Search Graph — {total_displayed}-Node View",
        fontsize=23,
        fontweight="bold",
        color="#0f172a",
        pad=28,
    )
    axis.text(
        0.5,
        1.01,
        (
            "Green path: START + 4 selected views  ·  "
            f"total route cost {route_cost:.2f} s"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#475569",
    )
    axis.text(
        0.5,
        0.02,
        (
            f"Actual graph subset: {len(displayed)} camera candidates + START. "
            f"Gray arrows show the {args.neighbors} lowest-cost outgoing edges "
            "per displayed candidate."
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#64748b",
    )
    axis.set_xlim(-3.85, 3.15)
    axis.set_ylim(-2.0, 2.05)
    axis.set_axis_off()
    figure.tight_layout(pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)

    print(args.output)
    print("selected route:", " -> ".join(route))
    print(
        "alternative candidates:",
        ", ".join(names[index] for index in alternatives),
    )


if __name__ == "__main__":
    render(parse_args())
