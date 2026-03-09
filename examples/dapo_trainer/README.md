# DAPO 训练示例

## Smoke Test（2 GPU 快速验证）

用 **Qwen3-2B base**（或 Qwen2.5-2B）在 **DAPO-17k** 上跑几步，确认环境能跑通。

### 1. 准备数据

DAPO-17k 使用 DAPO-Math-17k 数据集，需为 **parquet** 路径。

**方式 A：从 HuggingFace 下载**

```bash
# 安装 huggingface_hub 后
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k --local-dir $HOME/data/DAPO-Math-17k
# 若目录内有 data/dapo-math-17k.parquet，则 TRAIN_FILE=$HOME/data/DAPO-Math-17k/data/dapo-math-17k.parquet
# 否则根据实际 parquet 路径设置 TRAIN_FILE
```

**方式 B：使用已有 parquet**

若你已有 `dapo-math-17k.parquet`，直接设置环境变量：

```bash
export TRAIN_FILE=/path/to/dapo-math-17k.parquet
```

### 2. 指定 GPU

只使用 GPU 0 和 1（默认已设为 0,1）：

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

### 3. 运行 Smoke Test

```bash
cd /home/li.12312/yining/verl  # 或你的 verl 仓库根目录

# 若数据在 $HOME/data/dapo-math-17k.parquet
bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh

# 或显式指定数据路径
DATA_DIR=/path/to/data TRAIN_FILE=/path/to/dapo-math-17k.parquet bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh
```

脚本会跑 **2 个 training steps** 后退出，用于检查 DAPO 流程是否正常（数据、reward、模型加载、2 GPU 分配等）。

### 4. 模型路径

默认使用 `Qwen/Qwen2.5-2B`（HuggingFace 2B 基座）。若要用 Qwen3-2B 或本地路径：

```bash
MODEL_PATH=Qwen/Qwen3-2B bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh
# 或
MODEL_PATH=/path/to/local/qwen3-2b bash examples/dapo_trainer/run_dapo_qwen3_2b_smoke_test.sh
```

### 5. 常见问题

- **数据列**：DAPO reward 需要数据中有 `prompt`、`data_source`（如 `math_dapo`）、以及 `reward_model.ground_truth`（或数据集里配置的 ground_truth 列）。
- **OOM**：若 2 卡显存不足，可在脚本里再减小 `train_prompt_bsz`、`n_resp_per_prompt`、`max_response_length`。
