#!/bin/bash
set -e

###############################################################################
## 模块：vLLM-Standalone 启动/停止脚本
## 功能：以“vLLM OpenAI-Compatible Server”形态单独拉起模型服务，
##       支持多卡 Tensor Parallel、后台常驻、日志落盘、一键启停。
## 适用范围
##   - 仅负责“vLLM 进程”生命周期；训练、推理分离场景。
##   - 模型权重需提前下载到本地 ${MODEL_PATH}。
## 关键设计
##   1. 自动根据 GPU 列表计算 --tensor-parallel-size。
##   2. 日志双写：控制台 + ${LOG_FILE}，方便 tail -f 实时观察。
##   3. 停止逻辑使用 pgrep/pkill，匹配 vllm.entrypoints.openai.api_server
##      特征串，可一次杀干净。
## 使用示例
##   ./vllm_ctl.sh start 0,1,2,3   # 4 卡并行，后台启动
##   ./vllm_ctl.sh stop            # 一键关闭
###############################################################################

# --------------- 以下为用户可改常量 ---------------

# 固定配置
MODEL_PATH="/code/S1-Parser/model_weight/"
MAX_MODEL_LEN="12288"
GPU_MEM_UTIL="0.85"
PORT="8000"
SERVED_MODEL_NAME="Recognition-CUDA2"
MAX_NUM_SEQS="32"
CONDA_ENV="S1-Parser"
CONDA_INIT_PATH="/root/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="./log"
LOG_FILE="${LOG_DIR}/vllm.log"
VLLM_PROCESS_KEY="vllm.entrypoints.openai.api_server"

# 用法说明
usage() {
  echo "用法: $0 {start|stop} [GPU设备]"
  echo "启动示例: $0 start 0          # 后台启动服务，日志存${LOG_FILE}"
  echo "关闭示例: $0 stop             # 一键杀掉所有vLLM进程"
  exit 1
}

# 日志记录（带时间戳，控制台+日志文件双输出）
log() {
  local msg="$1"
  local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$timestamp] $msg" | tee -a "${LOG_FILE}"
}

# 启动vLLM服务（后台运行，释放命令行）
start_service() {
  local GPU_DEVICES="${1:?请指定GPU设备（如：0 或 0,2）}"

  # 创建日志目录
  if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
    log "日志目录已创建: ${LOG_DIR}"
  fi

  # 计算Tensor并行数
  local TENSOR_PARALLEL=$(echo "${GPU_DEVICES}" | tr -cd ',' | wc -c | awk '{print $1 + 1}')

  # 记录启动参数
  log "==================== 开始后台启动vLLM服务 ===================="
  log "模型路径: ${MODEL_PATH}"
  log "GPU设备: ${GPU_DEVICES}"
  log "最大模型长度: ${MAX_MODEL_LEN}"
  log "GPU显存利用率: ${GPU_MEM_UTIL}"
  log "服务端口: ${PORT}"
  log "服务模型名: ${SERVED_MODEL_NAME}"
  log "最大并发序列数: ${MAX_NUM_SEQS}"
  log "Tensor并行数: ${TENSOR_PARALLEL}"
  log "======================================================"

  # 加载Conda并激活环境
  if [ -f "${CONDA_INIT_PATH}" ]; then
    log "加载Conda初始化脚本: ${CONDA_INIT_PATH}"
    source "${CONDA_INIT_PATH}"
    log "激活conda环境: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}" || {
      log "错误：激活${CONDA_ENV}环境失败"
      exit 1
    }
  else
    log "错误：未找到Conda初始化脚本"
    exit 1
  fi

  # 后台启动vLLM（输出重定向到日志）
  log "开始后台启动vLLM OpenAI API服务..."
  CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --disable-custom-all-reduce \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --use-v2-block-manager \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    >"${LOG_FILE}" 2>&1 &

  log "服务已后台启动，日志见 ${LOG_FILE}，可执行 'tail -f ${LOG_FILE}' 实时查看"
}

# 关闭vLLM服务（一键杀所有进程）
stop_service() {
  if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p "${LOG_DIR}"
    log "日志目录已创建: ${LOG_DIR}"
  fi

  log "==================== 开始关闭vLLM服务 ===================="
  log "进程匹配特征: ${VLLM_PROCESS_KEY}"

  local VLLM_PIDS=$(pgrep -f "${VLLM_PROCESS_KEY}" 2>/dev/null || true)
  if [ -n "${VLLM_PIDS}" ]; then
    log "找到vLLM进程ID: ${VLLM_PIDS}，开始终止"
    pkill -f "${VLLM_PROCESS_KEY}" 2>/dev/null

    sleep 2
    local REMAIN_PIDS=$(pgrep -f "${VLLM_PROCESS_KEY}" 2>/dev/null || true)
    if [ -z "${REMAIN_PIDS}" ]; then
      log "所有vLLM进程已成功终止"
    else
      log "警告：残留进程ID: ${REMAIN_PIDS}（建议手动执行 kill -9 ${REMAIN_PIDS}）"
    fi
  else
    log "未找到vLLM进程"
  fi
  log "======================================================"
}

# 主逻辑
case "$1" in
start)
  shift
  start_service "$@"
  ;;
stop)
  stop_service
  ;;
*)
  usage
  ;;
esac
