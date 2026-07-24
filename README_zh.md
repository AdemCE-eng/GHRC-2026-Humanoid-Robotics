# 全球人形机器人挑战赛 2026 Baseline

全球人形机器人挑战赛 2026（Global Humanoid Robot Challenge 2026）官方技术文档，基于 [LeRobot](https://github.com/huggingface/lerobot) 框架构建的人形机器人仿真平台，提供物理仿真、数据采集、模型训练到部署的完整工作流。

## 项目概述

本文档面向 **全球人形机器人挑战赛 2026 (GHRC 2026)** 参赛者与研发团队，提供统一的基线实现：

- 基于 **NVIDIA Isaac Sim** 搭建高保真机器人仿真环境
- 通过键盘遥操作完成数据录制，输出标准化 **LeRobotDataset V3.0** 格式
- 基于模仿学习算法（ACT、SmolVLA、Pi0 等）进行模型训练与微调

## 核心能力

| 能力 | 说明 |
| :--: | --- |
| **仿真环境** | 基于 NVIDIA Isaac Sim 的高保真 Walker S2 机器人仿真，支持 20 维状态空间（14 臂关节 + 4 夹爪关节 + 2 夹持器控制指令） |
| **数据采集** | 支持键盘遥操作；输出 **LeRobotDataset V3.0** 格式 |
| **模型训练** | 支持 **ACT**、**Pi0** 等模仿学习算法 |
| **四目实时显示** | 支持 4 个 RGB 相机实时预览（head_left, head_right, wrist_left, wrist_right） |

## 资源说明

本项目部分大文件托管于 Hugging Face，首次使用前 **请先完成下载**：

| 资源类别 | 本地目录 | 远程地址 |
| --- | --- | --- |
| 🤖 仿真环境与机器人资产 | `assets/`（Git 子模块） | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets) |
| 📊 训练数据集 | `datasets/` | [UBTECH-Robotics/challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) |

### 配置信息

| | 最低要求 | 推荐配置 | 理想配置 |
| --- | --- | --- | --- |
| **操作系统** | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 |
| **CPU** | Intel Core i7 (7th Generation); AMD Ryzen 5 | Intel Core i7 (9th Generation); AMD Ryzen 7 | Intel Core i9, X-series or higher; AMD Ryzen 9, Threadripper or higher |
| **核心数** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **存储空间** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **驱动** | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 |

> 建议配置更大容量的 RAM 和 VRAM；特别是在需要训练模型的时候。另外如果安装的是 595 的驱动，可能后续导致 Docker 容器内运行 Isaac Sim 闪退的问题，因此推荐安装 580 的驱动。

### 工具要求

| 工具 | 版本 | 备注 |
| --- | --- | --- |
| `CUDA` | 12.8 | [官方指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) |
| `Docker` | latest | [官方指南](https://docs.docker.com/engine/install/ubuntu/) |
| `NVIDIA Container Toolkit` | latest | [官方指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Hugging Face` | latest | `pip install huggingface-hub`; `huggingface-cli --help`（验证安装） |
| `Git` | latest | `sudo apt update`; `sudo apt install git -y`; `git --version`（验证版本） |
| `Miniconda` | latest | [官方指南](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)（可选） |

## 技术文档索引

基线完整工作流分为七个阶段：**建议按顺序阅读**；也可根据实际进度直接跳转到所需阶段。

| # | 文档 | 说明 |
|---|------|------|
| 1 | [资源下载](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/1/) | 项目概述、核心能力、硬件配置要求、工具要求及 HuggingFace 资源链接。 |
| 2 | [环境配置](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/2/) | 拉取仓库代码、下载仿真资产与数据集、构建 Docker 镜像、配置键盘 evdev 设备。 |
| 3 | [快速开始](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/3/) | 启动运行环境、键盘遥操作、数据采集、数据回放及按键映射说明。 |
| 4 | [模型训练](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/4/) | ACT、Diffusion Policy、π₀ (PI0)、π₀.₅ (PI05)、SmolVLA 等策略的完整训练指南与超参数说明。 |
| 5 | [策略推理](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/5) | 使用训练好的策略模型进行推理并自动录制结果。 |
| 6 | [四目实时显示](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/6/) | 4 个 RGB 相机的实时预览与可视化配置，支持遥操作、数据采集、回放和推理四种模式。 |
| 7 | [自动数据采集工具](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/7/) | 通过自动化数据采集脚本进行任务数据采集。 |
