"""自动数采包 —— 编程式机器人数据采集框架。

基于模板方法模式：基类提供通用机器人控制接口和完整 episode 流水线，
任务子类只需实现 ``compute_grasp_poses``、``check_grasp_success`` 和
``get_place_pose`` 三个差异化接口。
"""

from .auto_collect_base import AutoCollectBase
from .auto_collect_config import AutoCollectConfig, AutoCollectMainConfig
from .task_conveyor_sorting import TaskConveyorSorting
from .task_part_sorting import TaskPartSorting
from .task_foam_inlaying import TaskFoamInlaying
from .task_packing_box import TaskPackingBox

__all__ = [
    "AutoCollectBase",
    "AutoCollectConfig",
    "AutoCollectMainConfig",
    "TaskConveyorSorting",
    "TaskPartSorting",
    "TaskFoamInlaying",
    "TaskPackingBox",
]
