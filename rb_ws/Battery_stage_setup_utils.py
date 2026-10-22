"""Stage setup helpers for Battery_rb.py."""

import os

import omni.usd
from pxr import UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils


def update_simulation_frames(
    simulation_app,
    update_count,
):
    for _ in range(update_count):
        simulation_app.update()


def open_stage_or_raise(
    simulation_app,
    usd_path,
):
    if not os.path.isfile(usd_path):
        simulation_app.close()
        raise FileNotFoundError(
            f"RB5 USD 파일을 찾을 수 없습니다: {usd_path}"
        )

    print("Opening RB5 stage:", usd_path)

    result = omni.usd.get_context().open_stage(
        usd_path
    )

    if not result:
        simulation_app.close()
        raise RuntimeError(
            f"RB5 USD Stage를 열지 못했습니다: {usd_path}"
        )

    return omni.usd.get_context().get_stage()


def print_relevant_stage_prims(
    stage,
    keywords,
):
    print("\n========== STAGE PRIMS ==========")

    lowered_keywords = [
        keyword.lower()
        for keyword in keywords
    ]

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_path_lower = prim_path.lower()

        if any(
            keyword in prim_path_lower
            for keyword in lowered_keywords
        ):
            print(prim_path)

    print("=================================\n")


def require_prim(
    stage,
    prim_path,
    description,
):
    prim = stage.GetPrimAtPath(
        prim_path
    )

    print(
        f"{prim_path} prim valid:",
        prim.IsValid(),
    )

    if not prim.IsValid():
        raise RuntimeError(
            f"{description} Prim을 찾지 못했습니다: {prim_path}"
        )

    return prim


def print_prim_children(
    prim,
    prim_path,
):
    print(
        f"{prim_path} children:",
        [str(child.GetPath()) for child in prim.GetChildren()],
    )


def find_articulation_roots(
    stage,
    robot_prim_path,
):
    articulation_roots = []

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        if (
            prim_path.startswith(robot_prim_path)
            and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            articulation_roots.append(prim_path)

    if articulation_roots:
        print("RB5 Articulation Root:", articulation_roots[0])
    else:
        print("WARNING: RB5 Articulation Root를 찾지 못했습니다.")
        print(
            "Expected root joint:",
            "/rb5_850e/Physics/root_joint",
        )

    return articulation_roots


def add_ground_plane(
    assets_root_path,
    simulation_app,
    update_count=10,
):
    stage_utils.add_reference_to_stage(
        usd_path=(
            assets_root_path
            + "/Isaac/Environments/Grid/default_environment.usd"
        ),
        path="/World/ground",
    )

    update_simulation_frames(
        simulation_app,
        update_count,
    )
