#!/usr/bin/env python
"""Approval-gated entrypoint for the Part Sorting Recovery Dataset V2 pilot."""

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

# Native Windows needs the existing project-local Pinocchio DLL bootstrap
# before importing the Walker S2 robot modules. This is a no-op in WSL/Linux.
if os.name == "nt":
    import Ubtech_sim.source  # noqa: F401

from src.lerobot.auto_collect.auto_collect_config import AutoCollectConfig
from src.lerobot.auto_collect.task_part_sorting_recovery_v2 import TaskPartSortingRecoveryV2
from src.lerobot.configs import parser
from src.lerobot.robots import make_robot_from_config
from src.lerobot.robots.config import RobotConfig
from src.lerobot.utils.utils import init_logging


@dataclass
class RecoveryV2CollectConfig(AutoCollectConfig):
    task: str = "Part_Sorting"
    repo_id: str = "local/part_sorting_recovery_v2_pilot"
    root: str | Path | None = None
    fps: int = 30
    num_episodes: int = 50
    record_data: bool = True
    push_to_hub: bool = False
    private: bool = True
    max_retries: int = 2
    objects_per_episode: int = 4
    random_speed: bool = False
    pilot_config: str | Path = Path("configs/recovery_v2_part_sorting_pilot.yaml")
    confirm_collection: bool = False
    stop_on_first_failure: bool = False
    stop_after_first_scenario_exhausted: bool = False

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.pilot_config, str):
            self.pilot_config = Path(self.pilot_config)


@dataclass
class RecoveryV2MainConfig:
    robot: RobotConfig
    auto_collect: RecoveryV2CollectConfig = field(default_factory=RecoveryV2CollectConfig)


@parser.wrap()
def auto_collect_recovery_v2(cfg: RecoveryV2MainConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    robot = make_robot_from_config(cfg.robot)
    if robot.name != "walkerS2" or not hasattr(robot, "config"):
        raise ValueError("Recovery V2 currently supports only walker_s2_sim")
    if robot.config.task_cfg_overrides:
        raise ValueError("Recovery V2 forbids task_cfg_overrides during the fixed pilot")
    robot.config.load_from_yaml("Part_Sorting")
    logging.info("Recovery V2 uses the unchanged Part_Sorting scene: %s", robot.config.task_cfg_path)
    TaskPartSortingRecoveryV2().run(robot, cfg.auto_collect)


if __name__ == "__main__":
    auto_collect_recovery_v2()
