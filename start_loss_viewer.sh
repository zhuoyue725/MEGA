#!/bin/bash
# Loss Viewer 启动脚本

echo "正在启动 Loss Viewer..."

# 检查Flask是否安装
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask未安装，正在安装依赖..."
    pip install -r loss_viewer_requirements.txt
fi

# 启动服务器
python3 loss_viewer.py --port 5000

# 如果需要使用其他端口，可以修改上面的 --port 参数
# 例如: python3 loss_viewer.py --port 8080
