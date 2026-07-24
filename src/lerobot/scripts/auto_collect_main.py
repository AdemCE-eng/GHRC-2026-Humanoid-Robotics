#!/usr/bin/env python
"""自动数采主入口 —— 编程式机器人数据采集。

用法::

    /isaac-sim/python.sh -m lerobot.scripts.auto_collect_main \\
        --robot.type=walker_s2_sim \\
        --auto_collect.task=Part_Sorting \\
        --auto_collect.repo_id=liberow/task1_collect \\
        --auto_collect.num_episodes=10 \\
        --auto_collect.record_data=true

支持的任务:
    - ``Part_Sorting`` (Task 1): 零件分类抓取放置
    - ``Conveyor_Sorting`` (Task 2): 传送带零件分拣
    - ``Foam_Inlaying`` (Task 3): 泡沫嵌入
"""

import json
import logging
from dataclasses import asdict
from pprint import pformat

from src.lerobot.auto_collect import (
    AutoCollectMainConfig,
    TaskConveyorSorting,
    TaskFoamInlaying,
    TaskPackingBox,
    TaskPartSorting,
)
from src.lerobot.configs import parser
from src.lerobot.robots import make_robot_from_config
from src.lerobot.utils.utils import init_logging

# 任务名称 → 采集器子类的注册表
_COLLECTOR_REGISTRY = {
    "Part_Sorting": TaskPartSorting,
    "Conveyor_Sorting": TaskConveyorSorting,
    "Foam_Inlaying": TaskFoamInlaying,
    "Packing_Box": TaskPackingBox,
}


def _deep_update(base: dict, overrides: dict) -> None:
    """递归地将 overrides 合并进 base（就地修改）。"""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def make_collector(task: str):
    """工厂函数：根据任务名返回对应的 ``AutoCollectBase`` 子类实例。"""
    cls = _COLLECTOR_REGISTRY.get(task)
    if cls is None:
        raise ValueError(
            f"未知任务 '{task}'。可用任务: {list(_COLLECTOR_REGISTRY.keys())}"
        )
    return cls()


@parser.wrap()
def auto_collect_main(cfg: AutoCollectMainConfig):
    """自动数采主入口函数。"""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # 在连接前加载任务 YAML 配置
    task = cfg.auto_collect.task
    if robot.name == "walkerS2" and hasattr(robot, "config"):
        robot.config.load_from_yaml(task)
        if robot.config.task_cfg_overrides:
            import ast
            try:
                overrides = json.loads(robot.config.task_cfg_overrides)
            except json.JSONDecodeError:
                overrides = ast.literal_eval(robot.config.task_cfg_overrides)
            _deep_update(robot.config.task_cfg, overrides)
            logging.info(f"[{robot.name}] 已应用 task_cfg_overrides: {overrides}")
        logging.info(f"[{robot.name}] 任务: {task}, 配置: {robot.config.task_cfg_path}")

    collector = make_collector(task)
    collector.run(robot, cfg.auto_collect)


if __name__ == "__main__":
    auto_collect_main()
