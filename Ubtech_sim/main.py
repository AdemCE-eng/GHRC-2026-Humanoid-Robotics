"""Ubtech_sim Simulation Entry Point.

Launch Isaac Sim, load task config, build scene, and run the grasp control loop.
"""
from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "headless": False,
}

kit = SimulationApp(launch_config=CONFIG)

# Isaac Sim modules must be imported after SimulationApp is created
from isaacsim.core.api import World
import omni
import omni.replicator.core as rep
import os
import numpy as np

from source.config_loader import load_config, apply_scatter_config
from source.SceneBuilder import SceneBuilder
from source.RobotArticulation import RobotArticulation
from source.DataLogger import DataLogger
from source.coordinate_utils import CoordinateTransform
from source.grasp_planner import GraspPlanner

# ── 1. Configuration ─────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "config/Part_Sorting.yaml")
cfg = load_config(config_path)
grasp_cfg = cfg.get("grasp", {})

# ── 2. Stage & World ─────────────────────────────────────────────────
omni.usd.get_context().open_stage(
    os.path.join(cfg["root_path"], cfg["scene_usd"])
)
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 20.0,
)
world.initialize_physics()

# ── 3. Data Logger ───────────────────────────────────────────────────
base_dir = os.path.dirname(__file__)
data_logger = DataLogger(
    enabled=True,
    csv_path=os.path.join(base_dir, "poses.csv"),
    camera_enabled=False,
    camera_hdf5_path=os.path.join(base_dir, "camera_data.hdf5"),
)

# ── 4. Scene (scatter area → build → physics settle) ────────────────
scene = SceneBuilder(cfg, data_logger=data_logger)
apply_scatter_config(cfg)

scene.build_all()
rep.orchestrator.step()
print("[Init] 场景物体已创建并 scatter，开始物理稳定...")

world.play()
settle_time = grasp_cfg.get("settle_time", 2.0)
settle_steps = int(settle_time / world.get_physics_dt())
for _ in range(settle_steps):
    world.step(render=False)
print(f"[Init] 物理稳定完成 ({settle_time}s, {settle_steps} steps)")

# ── 5. Query Part Poses ──────────────────────────────────────────────
part_poses = scene.get_parts_world_poses()
print(f"[Init] 查询到 {len(part_poses)} 个零件")
for pp in part_poses:
    print(f"  {pp['prim_path']}: pos={pp['position']}")

# ── 6. Robot ─────────────────────────────────────────────────────────
world.pause()
scene.build_robot()
robot = RobotArticulation(prim_path="/Root/Ref_Xform/Ref", name="walkerS2")
robot.initialize()
print("[Init] 机器人已加入场景并设置初始关节角（物理暂停中）")
world.play()

for _ in range(10):
    world.step(render=False)

# ── 7. IK & Coordinate Transform ────────────────────────────────────
urdf_path = os.path.join(cfg["root_path"], "s2.urdf")
robot.initialize_ik(urdf_path)

js = robot.get_joint_states()
if js is not None:
    robot.ik_solver.sync_joint_positions(js["names"], js["positions"][0])

compensation_matrix = np.array([
    [9.99999e-01, -1.11400e-03,  1.16200e-03, -9.64000e-04],
    [-2.00000e-05,  7.13609e-01,  7.00544e-01, -9.59927e-01],
    [-1.61000e-03, -7.00544e-01,  7.13608e-01,  6.56540e-01],
    [0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]
], dtype=np.float64)

coord_transform = CoordinateTransform.from_torso_link(ik_solver=robot.ik_solver,compensation_matrix=compensation_matrix)
for _ in range(10):
    coord_transform.verify_ee_alignment(robot.ik_solver)

# ── 8. Grasp Planning ───────────────────────────────────────────────
planner = GraspPlanner(grasp_cfg, robot, coord_transform)
planner.compute_grasp_target(part_poses)

# ── 9. Callbacks ─────────────────────────────────────────────────────
def robot_control_callback(step_size):
    planner.update_active_target()
    left_target, right_target, rot_weight = planner.get_control_targets()
    robot.control_dual_arm_ik(
        step_size,
        left_target_xyzrpy=left_target,
        right_target_xyzrpy=right_target,
        rot_weight=rot_weight,
    )
    # planner.log_debug()


def score_input_record_callback(step_size):
    scene.get_target_object_transforms(step_size)


def camera_images_callback(step):
    camera_data = robot.get_cameras_images(step)
    data_logger.log_camera_rgb(camera_data)


world.add_physics_callback("robot_control", robot_control_callback)
world.add_physics_callback("score_input_record", score_input_record_callback)
world.add_physics_callback("foam_sync", lambda dt: scene.sync_foam_to_box())
world.add_render_callback("camera_images", camera_images_callback)

# ── 10. Main Loop ────────────────────────────────────────────────────
try:
    while kit.is_running():
        world.step()
finally:
    data_logger.close()