from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


SDK_NATIVE_FRAME = "link6"


@dataclass(frozen=True)
class FrameSpec:
    urdf_path: Path
    native_frame: str
    tcp_frame: str
    T_native_to_tcp: np.ndarray
    T_tcp_to_native: np.ndarray


def resolve_tcp_frame(
    urdf_path: str | Path,
    tcp_frame: str,
    native_frame: str = SDK_NATIVE_FRAME,
) -> FrameSpec:
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()

    link_names = {link.attrib["name"] for link in root.findall("link")}
    if native_frame not in link_names:
        raise ValueError(f"Native frame '{native_frame}' not found in URDF: {urdf_path}")
    if tcp_frame not in link_names:
        raise ValueError(f"TCP frame '{tcp_frame}' not found in URDF: {urdf_path}")

    if tcp_frame == native_frame:
        identity = np.eye(4)
        return FrameSpec(urdf_path, native_frame, tcp_frame, identity, identity)

    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue

        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        parent_link = parent.attrib["link"]
        child_link = child.attrib["link"]
        origin = joint.find("origin")
        xyz = _parse_xyz_rpy(origin, "xyz")
        rpy = _parse_xyz_rpy(origin, "rpy")
        T_parent_to_child = np.eye(4)
        T_parent_to_child[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T_parent_to_child[:3, 3] = xyz

        graph.setdefault(parent_link, []).append((child_link, T_parent_to_child))
        graph.setdefault(child_link, []).append((parent_link, np.linalg.inv(T_parent_to_child)))

    queue = deque([(native_frame, np.eye(4))])
    visited = {native_frame}
    while queue:
        frame_name, T_native_to_frame = queue.popleft()
        if frame_name == tcp_frame:
            return FrameSpec(
                urdf_path=urdf_path,
                native_frame=native_frame,
                tcp_frame=tcp_frame,
                T_native_to_tcp=T_native_to_frame,
                T_tcp_to_native=np.linalg.inv(T_native_to_frame),
            )

        for neighbor_name, T_frame_to_neighbor in graph.get(frame_name, []):
            if neighbor_name in visited:
                continue
            visited.add(neighbor_name)
            queue.append((neighbor_name, T_native_to_frame @ T_frame_to_neighbor))

    raise ValueError(
        "No fixed-joint path found between "
        f"native frame '{native_frame}' and TCP frame '{tcp_frame}' in URDF: {urdf_path}"
    )


def pose_from_native(T_world_native: np.ndarray, frame_spec: FrameSpec) -> np.ndarray:
    return T_world_native @ frame_spec.T_native_to_tcp


def pose_to_native(T_world_tcp: np.ndarray, frame_spec: FrameSpec) -> np.ndarray:
    return T_world_tcp @ frame_spec.T_tcp_to_native


def _parse_xyz_rpy(origin: ET.Element | None, key: str) -> np.ndarray:
    if origin is None:
        return np.zeros(3)
    raw_value = origin.attrib.get(key, "0 0 0")
    return np.array([float(value) for value in raw_value.split()], dtype=float)