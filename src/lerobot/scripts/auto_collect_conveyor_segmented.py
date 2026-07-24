#!/usr/bin/env python
"""Standalone entry for segmented Conveyor Sorting auto collection.

This script leaves the original registry untouched and directly runs the new
collector that produces:

1. ``short_episodes``: one part per episode;
2. ``long_episodes``: one eight-part sequence per episode.
"""

import logging
from dataclasses import asdict
from pprint import pformat

from src.lerobot.auto_collect.auto_collect_config import AutoCollectMainConfig
from src.lerobot.auto_collect.task_conveyor_sorting_segmented import (
    TaskConveyorSortingSegmented,
)
from src.lerobot.configs import parser
from src.lerobot.robots import make_robot_from_config
from src.lerobot.utils.utils import init_logging


@parser.wrap()
def auto_collect_conveyor_segmented(cfg: AutoCollectMainConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    task = cfg.auto_collect.task
    if robot.name == "walkerS2" and hasattr(robot, "config"):
        robot.config.load_from_yaml(task)
        logging.info("[%s] 任务: %s, 配置: %s", robot.name, task, robot.config.task_cfg_path)

    collector = TaskConveyorSortingSegmented()
    collector.run(robot, cfg.auto_collect)


if __name__ == "__main__":
    auto_collect_conveyor_segmented()
