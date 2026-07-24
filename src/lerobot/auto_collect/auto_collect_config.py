"""自动数采配置 dataclass。

格式参考 ``lerobot_record.py`` 中的 ``RecordConfig``。
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.lerobot.robots.config import RobotConfig


@dataclass
class AutoCollectConfig:
    """编程式自动数采控制配置。

    CLI 使用示例::

        /isaac-sim/python.sh -m lerobot.scripts.auto_collect_main \\
            --robot.type=walker_s2_sim \\
            --auto_collect.task=Part_Sorting \\
            --auto_collect.repo_id=liberow/task1_collect \\
            --auto_collect.num_episodes=10 \\
            --auto_collect.record_data=true
    """

    # 任务名称: "Part_Sorting", "Conveyor_Sorting", "Foam_Inlaying", "Packing_Box"
    task: str = "Part_Sorting"

    # 数据集标识（格式: {hf_username}/{dataset_name}）
    repo_id: str = ""

    # 数据集存储根目录，None 时默认 ``$HF_LEROBOT_HOME/repo_id``
    root: str | Path | None = None

    # 控制频率 (帧/秒)
    fps: int = 30

    # 录制的 episode 数量
    num_episodes: int = 1

    # 是否录制数据到 LeRobotDataset
    record_data: bool = True

    # 录制完成后是否推送到 Hugging Face Hub
    push_to_hub: bool = False

    # 是否推送到私有仓库
    private: bool = False

    # Hub 数据集标签
    tags: list[str] | None = None

    # 是否将帧编码为视频
    video: bool = True

    # 视频编码。默认使用 h264，兼容性优先；可设为 auto 使用硬件编码，或 libsvtav1 获取更高压缩率。
    vcodec: str = "h264"

    # Number of episodes to record before batch encoding videos.
    video_encoding_batch_size: int = 1

    # Enable streaming video encoding during capture instead of writing PNG frames first.
    streaming_encoding: bool = False

    # Maximum number of frames to buffer per camera when streaming encoding is enabled.
    encoder_queue_maxsize: int = 30

    # Number of threads per encoder instance. None lets the codec decide.
    encoder_threads: int | None = None

    # 任务描述（None 时自动使用 task 名称）
    single_task: str | None = None

    # 单 episode 抓取失败最大重试次数
    max_retries: int = 20

    # 每 episode 最大物体数（0 = 处理场景中所有物体）
    objects_per_episode: int = 0

    # Arm execution order for tasks that support both sequential and simultaneous arms.
    # Supported by Foam_Inlaying: "dual", "left_then_right", "right_then_left".
    arm_execution_mode: str = "dual"


    # Segmented Conveyor Sorting failure handling mode:
    # "strict" = failed part makes the long episode fail;
    # "loosen" / "skip_failed_parts" = skip failed parts and continue.
    segmented_failure_mode: str = "loosen"

    # 写入 PNG 图像的子进程数
    num_image_writer_processes: int = 8

    # 每相机写入 PNG 图像的线程数
    num_image_writer_threads_per_camera: int = 4

    # push_to_hub 时在仓库中的子目录
    path_in_repo: str | None = None

    # 是否启用随机速度缩放：True 时每个 episode 从 [speed_scale_min, speed_scale_max] 均匀随机采样。
    # 1.0 = 原速，<1.0 加快，>1.0 减慢。两端相同时退化为固定速度。
    random_speed: bool = False
    speed_scale_min: float = 0.5
    speed_scale_max: float = 2.0

    # ---- replay 模式参数（Task 4 Packing_Box） ----
    # 源数据集标识（如 "packing_box_episode_50"）
    source_repo_id: str = "packing_box_episode_50"
    # 源数据集根目录
    source_root: str | Path | None = "/data/SJJ/UBT/lerobot_0.5.1/datasets/Packing_Box"
    # 要回放的源 episode 索引
    source_episode: int = 0
    # 帧采样间隔，每隔 N 帧取一帧回放（1 = 逐帧回放）
    source_step_interval: int = 1

    def __post_init__(self):
        if self.root is not None and isinstance(self.root, str):
            self.root = Path(self.root)


@dataclass
class AutoCollectMainConfig:
    """顶层配置，由 CLI parser 解析。"""

    robot: RobotConfig
    auto_collect: AutoCollectConfig = field(default_factory=AutoCollectConfig)
