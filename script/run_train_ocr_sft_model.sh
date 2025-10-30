#!/bin/bash

PROGRESS_NAME="./train/train_ocr_sft_model.py"

#1、获取当前时间
DATE=$(date '+%Y%m%d')
#echo $DATE

#删除缓存
find . -type d -name '__pycache__' -exec rm -r {} +

#创建日志目录
DIRECTORY="./log/"


#LOGDIR="$DIRECTORY/$DATE.log"
LOGDIR="$DIRECTORY/${DATE}_sft.log"

#启动新程序
export CUDA_VISIBLE_DEVICES=1,2,3
nohup python3 $PROGRESS_NAME >$LOGDIR 2>&1 &
