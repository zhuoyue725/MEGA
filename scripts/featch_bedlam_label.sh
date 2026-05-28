#!/bin/bash

# BEDLAM 标签下载脚本
# 使用方法: ./featch_bedlam_label.sh <username> <password>

if [ $# -ne 2 ]; then
    echo "错误: 需要提供用户名和密码"
    echo "使用方法: $0 <username> <password>"
    exit 1
fi

username=$1
password=$2

set -e  # 任何命令失败时退出

echo "创建目录..."
mkdir -p data/training_labels

echo "下载 BEDLAM 标签..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/all_npz_12_training.zip' -O './data/training_labels/all_npz_12_training.zip' --no-check-certificate --continue

echo "解压文件..."
unzip data/training_labels/all_npz_12_training.zip -d data/bedlam_labels

echo "完成！"