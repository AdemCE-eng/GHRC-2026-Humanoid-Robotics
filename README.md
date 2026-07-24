# Global Humanoid Robot Challenge 2026 Baseline

This is the official technical documentation for Global Humanoid Robot Challenge 2026 (GHRC 2026). It is built on the [LeRobot](https://github.com/huggingface/lerobot) framework and provides an end-to-end workflow for a humanoid robot simulation platform, covering physics simulation, data collection, model training, and deployment.

## Project Overview

This documentation is intended for **GHRC 2026** participants and R&D teams, providing a unified baseline implementation:

- Build a high-fidelity humanoid robot simulation environment based on **NVIDIA Isaac Sim**
- Collect data via keyboard teleoperation and export in the standardized **LeRobotDataset V3.0** format
- Train and fine-tune models using imitation learning algorithms (e.g., ACT, SmolVLA, Pi0)

## Key Capabilities

| Capability | Description |
| :--: | --- |
| **Simulation Environment** | High-fidelity Walker S2 robot simulation based on NVIDIA Isaac Sim, supporting a 20-dimensional state space (14 arm joints + 4 gripper joints + 2 gripper control commands). |
| **Data Collection** | Supports keyboard teleoperation; exports **LeRobotDataset V3.0** format. |
| **Model Training** | Supports imitation learning algorithms such as **ACT** and **Pi0**. |
| **4-View Real-time Display** | Supports real-time preview from 4 RGB cameras (head_left, head_right, wrist_left, wrist_right). |

## Resources

Some large files in this project are hosted on Hugging Face. **Please download them before first use**:

| Resource Type | Local Directory | Remote |
| --- | --- | --- |
| 🤖 Simulation environment & robot assets | `assets/` (Git submodule) | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets) |
| 📊 Training dataset | `datasets/` | [UBTECH-Robotics/challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) |

### Recommended Configuration

|  | Minimum | Recommended | Ideal |
| --- | --- | --- | --- |
| **OS** | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 |
| **CPU** | Intel Core i7 (7th Gen); AMD Ryzen 5 | Intel Core i7 (9th Gen); AMD Ryzen 7 | Intel Core i9 (X-series or higher); AMD Ryzen 9 / Threadripper (or higher) |
| **Cores** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **Storage** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **Driver** | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 |

> We recommend using larger RAM and VRAM capacities, especially for model training. Also, if you installed the 595 driver, Isaac Sim may crash inside the Docker container later; therefore, we recommend using driver version 580.

### Tool Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| `CUDA` | 12.8 | [Official Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) |
| `Docker` | latest | [Official Guide](https://docs.docker.com/engine/install/ubuntu/) |
| `NVIDIA Container Toolkit` | latest | [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Hugging Face` | latest | `pip install huggingface-hub`; `huggingface-cli --help` (verify installation) |
| `Git` | latest | `sudo apt update`; `sudo apt install git -y`; `git --version` (verify version) |
| `Miniconda` | latest | [Official Guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) (optional) |

## Technical Documentation Index

The complete baseline workflow consists of seven stages. **We recommend following the documents in order**; you may also jump directly to the stage you need based on your current progress.

| # | Document | Description |
|---|----------|-------------|
| 1 | [Resource Download](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/1/) | Project overview, key capabilities, hardware requirements, tool requirements, and HuggingFace resource links. |
| 2 | [Environment Setup](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/2/) | Clone the repository, download simulation assets and datasets, build the Docker image, and configure keyboard evdev. |
| 3 | [Quick Start](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/3/) | Start the runtime environment, keyboard teleoperation, data recording, dataset replay, and key mappings. |
| 4 | [Model Training](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/4/) | Training guides for ACT, Diffusion Policy, π₀ (PI0), π₀.₅ (PI05), and SmolVLA with full hyperparameters. |
| 5 | [Policy Inference](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/5/) | Run inference with a trained policy model and automatically record the results. |
| 6 | [4-Camera Real-time Display](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/6/) | Real-time preview from 4 RGB cameras and visualization configuration in teleoperation, recording, replay, and inference modes. |
| 7 | [Automated Data Collection Tool](https://docs.ubtrobot.com/GHRC2026_TechnicalDocuments/docs/7/) | Collect task data through automated data collection scripts. |
