#!/usr/bin/env python3
"""
Loss Viewer - 实时显示训练loss曲线的Web服务
使用方法: python loss_viewer.py --port 5000
"""
import os
import argparse
from flask import Flask, render_template_string, send_file
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# 默认checkpoint基础路径
CHECKPOINT_BASE = "/home/zzb/pydata/recons/MEGA/checkpoint/CVQDIFFUSION"

def find_latest_loss_image(base_path=CHECKPOINT_BASE):
    """
    在checkpoint目录中查找最新的loss.png文件
    遍历所有日期和时间文件夹，找到最新修改的loss.png
    """
    base = Path(base_path)
    if not base.exists():
        return None
    
    # 查找所有loss.png文件
    loss_files = list(base.glob("*/*/loss.png"))
    
    if not loss_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(loss_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)

def find_latest_reprojection_image(base_path=CHECKPOINT_BASE):
    """
    在checkpoint目录中查找最新的reprojection图片
    格式: xx_reprojection.png
    """
    base = Path(base_path)
    if not base.exists():
        return None
    
    # 查找所有reprojection图片
    reprojection_files = list(base.glob("*/*/samples_train/*_reprojection.png"))
    
    if not reprojection_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(reprojection_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)

def find_latest_compare_image(base_path=CHECKPOINT_BASE):
    """
    在checkpoint目录中查找最新的compare图片
    格式: epoch*_step*_cmp-compare.png
    """
    base = Path(base_path)
    if not base.exists():
        return None
    
    # 查找所有compare图片
    compare_files = list(base.glob("*/*/samples_train/epoch*_step*_cmp-compare.png"))
    
    if not compare_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(compare_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)

# 默认图像路径（自动查找最新的）
LOSS_IMAGE_PATH = find_latest_loss_image() or "/home/zzb/pydata/recons/MEGA/checkpoint/CVQDIFFUSION/2026-03-14/16-21/loss.png"
REPROJECTION_IMAGE_PATH = find_latest_reprojection_image()
COMPARE_IMAGE_PATH = find_latest_compare_image()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Monitor Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            max-width: 1400px;
            margin: 0 auto;
            backdrop-filter: blur(10px);
        }
        
        h1 {
            color: #2d3748;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 700;
        }
        
        .info {
            text-align: center;
            color: #4a5568;
            margin-bottom: 25px;
            font-size: 0.95em;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 15px 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            color: white;
            font-weight: 500;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            background: #48ff48;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .image-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        
        .image-container h2 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            transition: transform 0.3s ease;
        }
        
        .image-container img:hover {
            transform: scale(1.02);
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 968px) {
            .grid-container {
                grid-template-columns: 1fr;
            }
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .refresh-interval {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        select {
            padding: 8px 15px;
            border-radius: 6px;
            border: 2px solid #667eea;
            font-size: 0.95em;
            cursor: pointer;
            background: white;
        }
        
        .footer {
            margin-top: 20px;
            text-align: center;
            color: white;
            font-size: 0.9em;
        }
        
        .not-available {
            color: #999;
            font-style: italic;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Training Monitor Dashboard</h1>
        <div class="info">
            <p>实时监控模型训练Loss曲线和样本输出</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>实时更新中</span>
            </div>
            <div class="status-item">
                <span>📁 训练时间: {{ training_time }}</span>
            </div>
            <div class="status-item">
                <span id="lastUpdate">最后更新: 加载中...</span>
            </div>
            <div class="status-item">
                <span id="refreshCounter">刷新次数: 0</span>
            </div>
        </div>
        
        <div class="info" style="margin-top: 15px; font-size: 0.85em; color: #666;">
            <p>📂 Loss文件路径: {{ loss_path }}</p>
        </div>
        
        {% if loss_path %}
        <div class="info" style="background: #f7fafc; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-family: monospace; font-size: 0.9em; color: #2d3748;">
            <strong>📂 训练路径:</strong> {{ loss_path }}
        </div>
        {% endif %}
        
        <!-- Loss曲线 -->
        <div class="image-container">
            <h2>📊 Loss Curve</h2>
            {% if has_loss %}
            <img id="lossImage" src="/loss_image?t={{ timestamp }}" alt="Loss Curve" />
            {% else %}
            <div class="not-available">Loss图像暂不可用</div>
            {% endif %}
        </div>
        
        <!-- Reprojection和Compare图像 -->
        <div class="grid-container">
            <div class="image-container">
                <h2>🎯 Reprojection</h2>
                {% if has_reprojection %}
                <img id="reprojectionImage" src="/reprojection_image?t={{ timestamp }}" alt="Reprojection" />
                {% else %}
                <div class="not-available">Reprojection图像暂不可用</div>
                {% endif %}
            </div>
            
            <div class="image-container">
                <h2>🔍 Compare</h2>
                {% if has_compare %}
                <img id="compareImage" src="/compare_image?t={{ timestamp }}" alt="Compare" />
                {% else %}
                <div class="not-available">Compare图像暂不可用</div>
                {% endif %}
            </div>
        </div>
        
        <div class="controls">
            <button onclick="refreshImages()">🔄 立即刷新</button>
            <div class="refresh-interval">
                <label for="interval">刷新间隔:</label>
                <select id="interval" onchange="changeInterval()">
                    <option value="2">2秒</option>
                    <option value="5" selected>5秒</option>
                    <option value="10">10秒</option>
                    <option value="30">30秒</option>
                    <option value="60">1分钟</option>
                </select>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>💡 提示: 图像会自动刷新，无需手动操作</p>
    </div>
    
    <script>
        let refreshCount = 0;
        let intervalId = null;
        let currentInterval = 5000; // 默认5秒
        
        function refreshImages() {
            const timestamp = new Date().getTime();
            
            // 刷新Loss图像
            const lossImg = document.getElementById('lossImage');
            if (lossImg) {
                lossImg.src = '/loss_image?t=' + timestamp;
            }
            
            // 刷新Reprojection图像
            const reprojImg = document.getElementById('reprojectionImage');
            if (reprojImg) {
                reprojImg.src = '/reprojection_image?t=' + timestamp;
            }
            
            // 刷新Compare图像
            const compareImg = document.getElementById('compareImage');
            if (compareImg) {
                compareImg.src = '/compare_image?t=' + timestamp;
            }
            
            refreshCount++;
            document.getElementById('refreshCounter').textContent = '刷新次数: ' + refreshCount;
            document.getElementById('lastUpdate').textContent = '最后更新: ' + new Date().toLocaleTimeString('zh-CN');
        }
        
        function changeInterval() {
            const select = document.getElementById('interval');
            currentInterval = parseInt(select.value) * 1000;
            
            // 清除旧的定时器
            if (intervalId) {
                clearInterval(intervalId);
            }
            
            // 设置新的定时器
            intervalId = setInterval(refreshImages, currentInterval);
        }
        
        // 初始化自动刷新
        intervalId = setInterval(refreshImages, currentInterval);
        
        // 页面加载时立即刷新一次
        window.onload = function() {
            refreshImages();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页面"""
    timestamp = datetime.now().timestamp()
    
    # 每次访问时重新查找最新的图像
    latest_loss = find_latest_loss_image()
    latest_reprojection = find_latest_reprojection_image()
    latest_compare = find_latest_compare_image()
    
    return render_template_string(
        HTML_TEMPLATE, 
        loss_path=latest_loss or LOSS_IMAGE_PATH,
        timestamp=timestamp,
        has_loss=latest_loss is not None,
        has_reprojection=latest_reprojection is not None,
        has_compare=latest_compare is not None
    )

@app.route('/loss_image')
def loss_image():
    """返回loss图像"""
    # 每次请求时查找最新的loss图像
    latest_loss = find_latest_loss_image()
    if latest_loss and os.path.exists(latest_loss):
        return send_file(latest_loss, mimetype='image/png')
    elif os.path.exists(LOSS_IMAGE_PATH):
        return send_file(LOSS_IMAGE_PATH, mimetype='image/png')
    else:
        return "Loss image not found", 404

@app.route('/reprojection_image')
def reprojection_image():
    """返回reprojection图像"""
    latest_reprojection = find_latest_reprojection_image()
    if latest_reprojection and os.path.exists(latest_reprojection):
        return send_file(latest_reprojection, mimetype='image/png')
    else:
        return "Reprojection image not found", 404

@app.route('/compare_image')
def compare_image():
    """返回compare图像"""
    latest_compare = find_latest_compare_image()
    if latest_compare and os.path.exists(latest_compare):
        return send_file(latest_compare, mimetype='image/png')
    else:
        return "Compare image not found", 404

def main():
    global LOSS_IMAGE_PATH, REPROJECTION_IMAGE_PATH, COMPARE_IMAGE_PATH
    
    parser = argparse.ArgumentParser(description='Loss Viewer - 实时显示训练loss曲线')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0)')
    parser.add_argument('--loss-path', type=str, default=LOSS_IMAGE_PATH, help='Loss图像路径')
    
    args = parser.parse_args()
    
    # 更新全局路径
    LOSS_IMAGE_PATH = args.loss_path
    
    print("=" * 60)
    print("🚀 Training Monitor Dashboard 启动中...")
    print("=" * 60)
    
    # 检查文件是否存在
    latest_loss = find_latest_loss_image()
    latest_reprojection = find_latest_reprojection_image()
    latest_compare = find_latest_compare_image()
    
    if latest_loss and os.path.exists(latest_loss):
        print(f"✅ 找到Loss文件: {latest_loss}")
        checkpoint_dir = os.path.dirname(latest_loss)
        print(f"📁 Checkpoint目录: {checkpoint_dir}")
    else:
        print(f"⚠️  警告: Loss文件不存在")
    
    if latest_reprojection and os.path.exists(latest_reprojection):
        print(f"✅ 找到Reprojection文件: {latest_reprojection}")
    else:
        print(f"⚠️  警告: Reprojection文件不存在")
    
    if latest_compare and os.path.exists(latest_compare):
        print(f"✅ 找到Compare文件: {latest_compare}")
    else:
        print(f"⚠️  警告: Compare文件不存在")
    
    print(f"🌐 服务地址: http://{args.host}:{args.port}")
    print(f"💻 本地访问: http://localhost:{args.port}")
    print(f"🔗 远程访问: http://<你的服务器IP>:{args.port}")
    print("=" * 60)
    print("💡 提示: 图像会自动查找最新版本并刷新")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
