#!/usr/bin/env bash
CONTAINER_ID=""
WORK_DIR="/workspace/S1-Parser/code"

LOG_FILE="/workspace/S1-Parser/code/log/$(date +%Y%m%d)_message.log"

# 第 1~3 步：在容器里启动脚本
docker exec -d "$CONTAINER_ID" bash -c "
    source /root/miniconda3/etc/profile.d/conda.sh&&
    conda activate bedrockocr &&
    cd $WORK_DIR &&
    ./run_ocr_literature_center_inference_app.sh
"

# 第 4 步：在宿主机实时打印容器内日志
sleep 1        
docker exec -it "$CONTAINER_ID" tail -n 200 -f "$LOG_FILE"

