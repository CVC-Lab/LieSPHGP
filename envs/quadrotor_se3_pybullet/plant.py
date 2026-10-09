"""PyBullet plant helpers shared by the dataset generator and the closed-loop tools.

    remove_ground_plane(env)               delete every non-drone body (ground plane) and check no contact remains
    configure_contact_free_dynamics(env)   zero all contact friction; Bullet's built-in damping on or off
    motor_mixer(env)                       [thrust, tau_x, tau_y, tau_z] = M [f_1..f_4] for the CF2P "+" frame, and M^-1

`env` is a gym_pybullet_drones CtrlAviary (only its CLIENT, DRONE_IDS, KF, KM and L are used).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as pb

DAMPING_COEFFICIENT = 0.5      # Bullet's built-in linearDamping / angularDamping when the "nonlinear" law is on


def contact_point_count(env) -> int:
    return len(pb.getContactPoints(bodyA=int(env.DRONE_IDS[0]), physicsClientId=int(env.CLIENT)))


def remove_ground_plane(env) -> dict[str, Any]:
    client = int(env.CLIENT)
    drone_ids = {int(body_id) for body_id in np.asarray(env.DRONE_IDS).reshape(-1)}
    before = [int(pb.getBodyUniqueId(index, physicsClientId=client)) for index in range(pb.getNumBodies(physicsClientId=client))]
    removed = []
    for body_id in before:
        if body_id in drone_ids:
            continue
        name = pb.getBodyInfo(body_id, physicsClientId=client)[1].decode("utf-8", errors="replace")
        removed.append({"body_id": body_id, "body_name": name})
        pb.removeBody(body_id, physicsClientId=client)
    env.PLANE_ID = -1
    after = [int(pb.getBodyUniqueId(index, physicsClientId=client)) for index in range(pb.getNumBodies(physicsClientId=client))]
    if set(after) != drone_ids:
        raise AssertionError(f"No-ground audit failed: remaining={after}, drones={drone_ids}")
    contacts = contact_point_count(env)
    if contacts != 0:
        raise AssertionError(f"No-ground audit failed: {contacts} contact points remain")
    return {
        "ground_plane_removed": any("plane" in item["body_name"].lower() for item in removed),
        "removed_non_drone_bodies": removed,
        "body_ids_before": before,
        "body_ids_after": after,
        "remaining_bodies_are_only_drones": True,
        "contact_points_after_removal": contacts,
        "collision_environment": "no ground plane and no obstacle bodies",
    }


def configure_contact_free_dynamics(env, damping_law: str = "nonlinear") -> dict[str, Any]:
    """Zero every contact friction; built-in nonlinear damping (default) or built-in damping off for the linear law."""
    if damping_law not in ("nonlinear", "linear"):
        raise ValueError(f"unknown damping_law {damping_law!r}")
    builtin = DAMPING_COEFFICIENT if damping_law == "nonlinear" else 0.0
    client = int(env.CLIENT)
    records = []
    for body_index in range(pb.getNumBodies(physicsClientId=client)):
        body_id = int(pb.getBodyUniqueId(body_index, physicsClientId=client))
        name = pb.getBodyInfo(body_id, physicsClientId=client)[1].decode("utf-8", errors="replace")
        for link_index in range(-1, pb.getNumJoints(body_id, physicsClientId=client)):
            pb.changeDynamics(
                body_id, link_index,
                lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0,
                linearDamping=0.0, angularDamping=0.0, physicsClientId=client,
            )
            info = pb.getDynamicsInfo(body_id, link_index, physicsClientId=client)
            records.append({
                "body_id": body_id, "body_name": name, "link_index": link_index,
                "collision_shape_count": len(pb.getCollisionShapeData(body_id, link_index, physicsClientId=client)),
                "lateral_friction": float(info[1]), "rolling_friction": float(info[6]), "spinning_friction": float(info[7]),
            })
    pb.changeDynamics(
        int(env.DRONE_IDS[0]), -1,
        linearDamping=builtin, angularDamping=builtin, physicsClientId=client,
    )
    if any(
        record[key] != 0.0
        for record in records if record["collision_shape_count"] > 0
        for key in ("lateral_friction", "rolling_friction", "spinning_friction")
    ):
        raise AssertionError("Contact-friction zero audit failed")
    return {
        "damping_law": damping_law,
        "linear_damping_coefficient": DAMPING_COEFFICIENT,
        "angular_damping_coefficient": DAMPING_COEFFICIENT,
        "pybullet_builtin_damping": damping_law == "nonlinear",
        "custom_external_linear_damping": damping_law == "linear",
        "contact_friction_disabled": True,
        "body_links": records,
    }


def motor_mixer(env) -> tuple[np.ndarray, np.ndarray]:
    ratio = env.KM / env.KF
    matrix = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [0.0, env.L, 0.0, -env.L],
        [-env.L, 0.0, env.L, 0.0],
        [-ratio, ratio, -ratio, ratio],
    ])
    return matrix, np.linalg.inv(matrix)
