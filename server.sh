#!/usr/bin/env bash
set -euo pipefail

# ==============================
# 基本路径与日志
# ==============================
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/server_${TS}.log"

# 运行产物目录
RUN_DIR="./runs/Cifar10/FedEXT/${TS}"   # 可按需替换
mkdir -p "${RUN_DIR}"

# ==============================
# Server 网络/设备
# ==============================
BIND="0.0.0.0:50052"
DEVICE="cuda"                 # "cuda" 或 "cpu"
SERVER_GPU_ID="0"             # 仅当 DEVICE=cuda 时生效，映射到 CUDA_VISIBLE_DEVICES

# 是否限制 CPU 线程到 1，降低 CPU 抢占
ENV_OMP1=1                    # 1=限制；0=不限制

# ==============================
# 数据/任务配置（与 launch.py 同参）
# ==============================
DATA_ROOT="./dataset"
DATASET="Cifar10"
NUM_CLASSES=10
NUM_CLIENTS=2
ROUNDS=1
LOCAL_EPOCHS=5
BATCH_SIZE=64
LR=0.01
MOMENTUM=0.9
SAMPLE_FRACTION=1.0
SEED=42
MODEL_NAME="resnet18"
FEATURE_DIM=512
MAX_MESSAGE_MB=256
ENCODER_RATIO=0.2             # FedEXT 的 encoder_ratio
ALGORITHM="FedEXT"            # "FedEXT" / "FedAvg"

# ==============================
# Feature 导出阶段配置（服务端下发）
# ==============================
FEATURE_BATCH_SIZE=128
FEATURE_KEEP_SPATIAL=1        # 1=让客户端保留空间特征；0=扁平化
FEATURE_INCLUDE_TEST=1        # 1=让客户端上传test特征；0=禁用（对应 --feature_no_test_split）

# ==============================
# Tail（服务端“尾部分类器”）训练配置
# ==============================
TAIL_BATCH_SIZE=64
TAIL_EPOCHS=2
TAIL_LR=0.01
TAIL_MOMENTUM=0.9
TAIL_WEIGHT_DECAY=1e-4
TAIL_DEVICE="cuda"            # 可设为 "cpu"；空则跟随 --device
TAIL_MODEL_NAME="resnet34"    # 用于重建 tail 模型的 backbone

# ==============================
# WandB（服务器端支持）
# ==============================
USE_WANDB=0
WANDB_PROJECT="fedktl"
WANDB_ENTITY="epicmo"
WANDB_RUN_NAME="server-${TS}"

# ==============================
# 环境变量
# ==============================
export PYTHONUNBUFFERED=1
# 避免 CUDA 显存碎片化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

if [[ "${DEVICE,,}" == "cuda" ]]; then
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUDA_VISIBLE_DEVICES="${SERVER_GPU_ID}"
else
  # 强制 CPU
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ "${ENV_OMP1}" == "1" ]]; then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

# ==============================
# 组装 WandB 参数
# ==============================
WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_entity "${WANDB_ENTITY}" --wandb_run_name "${WANDB_RUN_NAME}")
fi

# ==============================
# 组装 Feature/Tail 开关
# ==============================
FEATURE_FLAGS=()
if [[ "${FEATURE_KEEP_SPATIAL}" == "1" ]]; then
  FEATURE_FLAGS+=(--feature_keep_spatial)
fi
if [[ "${FEATURE_INCLUDE_TEST}" == "0" ]]; then
  FEATURE_FLAGS+=(--feature_no_test_split)
fi

TAIL_ARGS=()
if [[ -n "${TAIL_DEVICE}" ]]; then
  TAIL_ARGS+=(--tail_device "${TAIL_DEVICE}")
fi
if [[ -n "${TAIL_MODEL_NAME}" ]]; then
  TAIL_ARGS+=(--tail_model_name "${TAIL_MODEL_NAME}")
fi

# ==============================
# 打印概览
# ==============================
echo "[Server] 启动中..."
echo "  dataset     : ${DATASET}"
echo "  clients     : ${NUM_CLIENTS}"
echo "  rounds      : ${ROUNDS}"
echo "  device      : ${DEVICE} (GPU_ID=${SERVER_GPU_ID})"
echo "  run_dir     : ${RUN_DIR}"
echo "  log         : ${LOG_FILE}"
echo "  feature     : keep_spatial=${FEATURE_KEEP_SPATIAL}, include_test=${FEATURE_INCLUDE_TEST}, batch_size=${FEATURE_BATCH_SIZE}"
echo "  tail        : bs=${TAIL_BATCH_SIZE}, epochs=${TAIL_EPOCHS}, lr=${TAIL_LR}, momentum=${TAIL_MOMENTUM}, wd=${TAIL_WEIGHT_DECAY}, device=${TAIL_DEVICE}, model=${TAIL_MODEL_NAME}"
if [[ "${USE_WANDB}" == "1" ]]; then
  echo "  wandb       : project=${WANDB_PROJECT}, run=${WANDB_RUN_NAME}"
else
  echo "  wandb       : disabled"
fi
echo

# ==============================
# 启动 Server（tee 同时输出到屏幕与文件）
# ==============================
python -m server.server_main \
  --bind "${BIND}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET}" \
  --num_classes "${NUM_CLASSES}" \
  --num_clients "${NUM_CLIENTS}" \
  --rounds "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --momentum "${MOMENTUM}" \
  --sample_fraction "${SAMPLE_FRACTION}" \
  --seed "${SEED}" \
  --model_name "${MODEL_NAME}" \
  --feature_dim "${FEATURE_DIM}" \
  --max_message_mb "${MAX_MESSAGE_MB}" \
  --encoder_ratio "${ENCODER_RATIO}" \
  --algorithm "${ALGORITHM}" \
  --device "${DEVICE}" \
  --run_dir "${RUN_DIR}" \
  --feature_batch_size "${FEATURE_BATCH_SIZE}" \
  --tail_batch_size "${TAIL_BATCH_SIZE}" \
  --tail_epochs "${TAIL_EPOCHS}" \
  --tail_lr "${TAIL_LR}" \
  --tail_momentum "${TAIL_MOMENTUM}" \
  --tail_weight_decay "${TAIL_WEIGHT_DECAY}" \
  "${FEATURE_FLAGS[@]}" \
  "${TAIL_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"