#!/usr/bin/env python3
"""Render a presentation-friendly summary of the Battery IMSI route graph."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from visualize_battery_imsi_networkx import load_graph_data


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
            / "Battery_imsi_networkx_graph_summary.png"
        ),
    )
    return parser.parse_args()


def short_pose_label(name: str) -> str:
    tilt_match = re.fullmatch(
        r"tilt_x_(pos|neg)_(\d+)_z_(pos|neg)_(\d+)",
        name,
    )
    if tilt_match:
        x_sign, x_angle, z_sign, z_angle = tilt_match.groups()
        return (
            f"Tilt X {'+' if x_sign == 'pos' else '−'}{x_angle}°\n"
            f"Z {'+' if z_sign == 'pos' else '−'}{z_angle}°"
        )
    local_match = re.fullmatch(r"local_x_(pos|neg)_(\d+)deg", name)
    if local_match:
        sign, angle = local_match.groups()
        return f"Local X {'+' if sign == 'pos' else '−'}{angle}°"
    return name.replace("_", " ")


def render(args: argparse.Namespace) -> None:
    names, edge_costs, initial_costs, route, _, summary = load_graph_data(
        args.cache_dir
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    route_indices = [name_to_index[name] for name in route]

    start = "START"
    route_nodes = [f"R{order}" for order in range(1, len(route) + 1)]
    other = "OTHER"
    graph = nx.DiGraph()
    graph.add_nodes_from([start, *route_nodes, other])
    selected_edges = list(zip([start, *route_nodes[:-1]], route_nodes))
    graph.add_edges_from(selected_edges)

    positions = {start: (-3.25, 0.0), other: (-0.55, 1.28)}
    route_x = np.linspace(-1.95, 2.65, len(route_nodes))
    positions.update(
        {node: (float(x), 0.0) for node, x in zip(route_nodes, route_x)}
    )

    figure, axis = plt.subplots(figsize=(16, 8.5), dpi=170)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    aggregate_edges = [(start, other)]
    aggregate_edges += [(other, node) for node in route_nodes]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=aggregate_edges,
        ax=axis,
        edge_color="#cbd5e1",
        width=1.5,
        alpha=0.85,
        style="dashed",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=13,
        connectionstyle="arc3,rad=0.06",
        node_size=2400,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=selected_edges,
        ax=axis,
        edge_color="#059669",
        width=5.0,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=24,
        connectionstyle="arc3,rad=0.0",
        node_size=2600,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[start],
        ax=axis,
        node_color="#fef3c7",
        edgecolors="#d97706",
        linewidths=2.7,
        node_size=2600,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=route_nodes,
        ax=axis,
        node_color="#dcfce7",
        edgecolors="#059669",
        linewidths=2.7,
        node_size=2600,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[other],
        ax=axis,
        node_color="#e2e8f0",
        edgecolors="#64748b",
        linewidths=2.2,
        node_shape="h",
        node_size=4200,
    )

    labels = {start: "Current\nstate"}
    labels.update(
        {
            node: f"View {order}\n{short_pose_label(name)}"
            for order, (node, name) in enumerate(
                zip(route_nodes, route),
                start=1,
            )
        }
    )
    labels[other] = f"Other feasible views\n× {len(names) - len(route)}"
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=10.5,
        font_weight="bold",
        font_color="#0f172a",
    )

    edge_cost_labels = {}
    for edge_index, edge in enumerate(selected_edges):
        if edge_index == 0:
            cost = float(initial_costs[route_indices[0]])
        else:
            source_index = route_indices[edge_index - 1]
            target_index = route_indices[edge_index]
            cost = float(edge_costs[source_index, target_index])
        edge_cost_labels[edge] = f"{cost:.2f} s"
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_cost_labels,
        ax=axis,
        font_size=10,
        font_color="#047857",
        font_weight="bold",
        label_pos=0.5,
        rotate=False,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "#ffffff",
            "edgecolor": "#a7f3d0",
            "alpha": 0.95,
        },
    )

    route_cost = float(summary["initial_plan"]["cost"])
    axis.set_title(
        "Motion-Aware NBV Graph — Simplified",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
        pad=28,
    )
    axis.text(
        0.5,
        1.01,
        (
            f"{len(names)} reachable candidates evaluated  →  "
            f"Beam Search selected a {len(route)}-view seed route "
            f"(total cost {route_cost:.2f} s)"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#475569",
    )
    axis.text(
        0.5,
        0.035,
        (
            f"The remaining {len(names) - len(route)} candidates are grouped "
            "into one node for presentation clarity. Dashed arrows represent "
            "alternative directed connections."
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#64748b",
    )
    axis.set_xlim(-3.8, 3.25)
    axis.set_ylim(-0.9, 2.0)
    axis.set_axis_off()
    figure.tight_layout(pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    render(parse_args())
