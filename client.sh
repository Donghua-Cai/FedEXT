#!/usr/bin/env bash
set -euo pipefail

# ==============================
# 基本路径与日志
# ==============================
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TS="$(date '+%Y%m%d_%H%M%S')"

# ==============================
# 连接/设备
# ==============================
SERVER_ADDR="127.0.0.1:50052"     # 与 server --bind 对应
CLIENT_NAME="client_$(hostname)"
DEVICE="cuda"                     # "cuda" / "cpu"
CLIENT_GPU_ID="0"                 # 仅当 DEVICE=cuda 时生效

# 是否限制 CPU 线程到 1
ENV_OMP1=1

# ==============================
# 数据/任务（与 client_main.py 同参）
# ==============================
DATA_ROOT="./dataset"
DATASET="Cifar10"
NUM_CLASSES=10
NUM_CLIENTS=2
BATCH_SIZE=64
SEED=42
FEATURE_DIM=512
ENCODER_RATIO=0.2
ALGORITHM="FedEXT"
MAX_MESSAGE_MB=256
NUM_WORKERS=       # 留空=默认；例如设 "4"

# 可选：客户端本地保存最终模型（当收到 finalize 且服务器下发最终模型时）
RUN_DIR_CLIENT=""  # 例如 "./client_runs/${CLIENT_NAME}/${TS}"

# ==============================
# 环境变量
# ==============================
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

if [[ "${DEVICE,,}" == "cuda" ]]; then
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUDA_VISIBLE_DEVICES="${CLIENT_GPU_ID}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ "${ENV_OMP1}" == "1" ]]; then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
fi

LOG_FILE="${LOG_DIR}/${CLIENT_NAME}.log"

echo "[Client] 启动中..."
echo "  name      : ${CLIENT_NAME}"
echo "  server    : ${SERVER_ADDR}"
echo "  dataset   : ${DATASET}"
echo "  device    : ${DEVICE} (GPU_ID=${CLIENT_GPU_ID})"
echo "  log       : ${LOG_FILE}"
if [[ -n "${RUN_DIR_CLIENT}" ]]; then
  echo "  save_to   : ${RUN_DIR_CLIENT}"
fi
echo

# ==============================
# 组装可选参数
# ==============================
OPT_ARGS=()
if [[ -n "${NUM_WORKERS}" ]]; then
  OPT_ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if [[ -n "${RUN_DIR_CLIENT}" ]]; then
  mkdir -p "${RUN_DIR_CLIENT}"
  OPT_ARGS+=(--run_dir "${RUN_DIR_CLIENT}")
fi

# ==============================
# 启动 Client（tee 同时输出到屏幕与文件）
# ==============================
python -m client.client_main \
  --server "${SERVER_ADDR}" \
  --client_name "${CLIENT_NAME}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET}" \
  --num_classes "${NUM_CLASSES}" \
  --num_clients "${NUM_CLIENTS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --feature_dim "${FEATURE_DIM}" \
  --encoder_ratio "${ENCODER_RATIO}" \
  --algorithm "${ALGORITHM}" \
  --max_message_mb "${MAX_MESSAGE_MB}" \
  --device "${DEVICE}" \
  "${OPT_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"