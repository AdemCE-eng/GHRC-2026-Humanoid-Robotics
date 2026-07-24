"""自动数采通用工具函数。"""

import logging
import numpy as np


def quaternion_to_rpy(quat_xyzw: np.ndarray) -> tuple[float, float, float]:
    """四元数 [x, y, z, w] 转换为 RPY (roll, pitch, yaw)，单位：弧度。"""
    x, y, z, w = quat_xyzw

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = float(np.copysign(np.pi / 2, sinp))
    else:
        pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def get_part_sorting_part_type(part: dict, scene_builder) -> str:
    """根据 prim 路径判断零件类型（Task 1 - Part Sorting）。

    Args:
        part: 包含 ``"prim_path"`` 的零件信息字典。
        scene_builder: SceneBuilder 实例，用于获取零件列表配置。

    Returns:
        ``"part_a"`` 或 ``"part_b"``。无法判定时默认返回 ``"part_b"``。
    """
    prim_path = part.get("prim_path", "")

    # 方式1: 路径前缀中包含 Part_A_ 或 Part_B_
    if "Part_A_" in prim_path or "part_a_" in prim_path:
        return "part_a"
    if "Part_B_" in prim_path or "part_b_" in prim_path:
        return "part_b"

    # 方式2: 使用 SceneBuilder 在创建/重置零件时记录的类型映射。
    if scene_builder is not None and hasattr(scene_builder, "part_type_by_prim_path"):
        part_type_by_prim_path = scene_builder.part_type_by_prim_path
        part_type = part_type_by_prim_path.get(prim_path)
        if part_type in ("part_a", "part_b"):
            return part_type

    # 方式3: 通过 prim_path 在 parts_prim_paths 中的索引判断
    if scene_builder is not None and hasattr(scene_builder, "parts_prim_paths"):
        parts_prim_paths = scene_builder.parts_prim_paths
        if prim_path in parts_prim_paths:
            idx = parts_prim_paths.index(prim_path)
            num_a = scene_builder.part_cfg.get("num_parts_a", scene_builder.part_cfg.get("num_parts", 2))
            return "part_a" if idx < num_a else "part_b"

    logging.warning(f"无法判断零件类型: {prim_path}，默认为 part_b")
    return "part_b"


def get_conveyor_sorting_part_type(part: dict, scene_builder=None) -> str:
    """根据 prim 路径判断零件类型（Task 2 - Conveyor Sorting）。

    Task 2 中 part_a 投放到左侧箱子，part_b 投放到右侧箱子。优先使用
    prim 路径中的 Part_A/Part_B 标识；缺少路径信息时，回退到索引交替规则，
    与当前 Task2 场景中 8 个刚体的默认生成顺序保持兼容。
    """
    prim_path = part.get("prim_path", "")

    if "Part_A" in prim_path or "part_a" in prim_path:
        return "part_a"
    if "Part_B" in prim_path or "part_b" in prim_path:
        return "part_b"

    part_index = part.get("index")
    if part_index is not None:
        return "part_a" if int(part_index) % 2 == 0 else "part_b"

    if scene_builder is not None and hasattr(scene_builder, "parts_prim_paths"):
        parts_prim_paths = scene_builder.parts_prim_paths
        if prim_path in parts_prim_paths:
            idx = parts_prim_paths.index(prim_path)
            return "part_a" if idx % 2 == 0 else "part_b"

    logging.warning(f"无法判断传送带零件类型: {prim_path}，默认为 part_b")
    return "part_b"


# 小嵌入孔对应的 Scene prim 路径（part_a，由 left 臂抓取）
_SMALL_HOLE_PRIM_PATHS = {
    "/Replicator/Ref_Xform_02",
    "/Replicator/Ref_Xform_03",
    "/Replicator/Ref_Xform_04",
}
# 大嵌入孔对应的 Scene prim 路径（part_b，由 right 臂抓取）
_LARGE_HOLE_PRIM_PATHS = {
    "/Replicator/Ref_Xform_05",
    "/Replicator/Ref_Xform_06",
    "/Replicator/Ref_Xform_07",
}


def get_foam_inlaying_part_type(part: dict, scene_builder=None) -> str:
    """根据 prim 路径判断零件类型（Task 3 - Foam Inlaying）。

    Task 3 中 part_a（小工件）由 left 臂抓取放入小嵌入孔，
    part_b（大工件）由 right 臂抓取放入大嵌入孔。

    通过 prim 路径直接匹配孔位 prim：
        - ``/Replicator/Ref_Xform_02/03/04`` → part_a（小工件）
        - ``/Replicator/Ref_Xform_05/06/07`` → part_b（大工件）

    Args:
        part: 包含 ``"prim_path"`` 的零件信息字典。
        scene_builder: 保留参数（兼容旧调用），不再使用。

    Returns:
        ``"part_a"``（小工件）或 ``"part_b"``（大工件）。
    """
    prim_path = part.get("prim_path", "")

    # 方式1: prim 路径直接匹配孔位
    if prim_path in _SMALL_HOLE_PRIM_PATHS:
        return "part_a"
    if prim_path in _LARGE_HOLE_PRIM_PATHS:
        return "part_b"

    # 方式2: 路径前缀中包含 Part_A_/Part_B_ 或 28motor
    if "Part_A_" in prim_path or "part_a_" in prim_path or "28motor" in prim_path:
        return "part_a"
    if "Part_B_" in prim_path or "part_b_" in prim_path:
        return "part_b"

    logging.warning(f"无法判断零件类型: {prim_path}，默认为 part_b")
    return "part_b"
