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

def get_latest_checkpoint_dir(base_path=CHECKPOINT_BASE):
    """
    获取最新的checkpoint文件夹路径 (日期/时间)
    """
    base = Path(base_path)
    if not base.exists():
        return None
    
    # 获取所有日期文件夹
    date_folders = sorted([d for d in base.iterdir() if d.is_dir()])
    if not date_folders:
        return None
    
    # 从最新的日期开始
    for date_folder in reversed(date_folders):
        time_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])
        if time_folders:
            # 返回最新的时间文件夹
            return time_folders[-1]
    
    return None

def find_latest_loss_image(base_path=CHECKPOINT_BASE):
    """
    在最新的checkpoint文件夹中查找loss_metrics.png
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return None
    
    loss_file = checkpoint_dir / "loss_metrics.png"
    if loss_file.exists():
        return str(loss_file)
    
    return None

def find_latest_reprojection_image(base_path=CHECKPOINT_BASE):
    """
    在最新的checkpoint文件夹中查找reprojection图片
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return None
    
    samples_dir = checkpoint_dir / "samples_train"
    if not samples_dir.exists():
        return None
    
    reprojection_files = list(samples_dir.glob("*_reprojection*.png"))
    if reprojection_files:
        # 返回最新修改的
        return str(max(reprojection_files, key=lambda p: p.stat().st_mtime))
    
    return None

def find_latest_compare_image(base_path=CHECKPOINT_BASE):
    """
    在最新的checkpoint文件夹中查找compare图片
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return None
    
    samples_dir = checkpoint_dir / "samples_train"
    if not samples_dir.exists():
        return None
    
    compare_files = list(samples_dir.glob("epoch*_step*_cmp-compare.png"))
    if compare_files:
        # 返回最新修改的
        return str(max(compare_files, key=lambda p: p.stat().st_mtime))
    
    return None

def get_all_reprojection_images(base_path=CHECKPOINT_BASE):
    """
    获取当前checkpoint文件夹中的所有reprojection图片，按修改时间排序
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return []
    
    samples_dir = checkpoint_dir / "samples_train"
    if not samples_dir.exists():
        return []
    
    reprojection_files = list(samples_dir.glob("*_reprojection*.png"))
    # 按修改时间排序（最新的在前）
    reprojection_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in reprojection_files]

def get_all_compare_images(base_path=CHECKPOINT_BASE):
    """
    获取当前checkpoint文件夹中的所有compare图片，按修改时间排序
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return []
    
    samples_dir = checkpoint_dir / "samples_train"
    if not samples_dir.exists():
        return []
    
    compare_files = list(samples_dir.glob("epoch*_step*_cmp-compare.png"))
    # 按修改时间排序（最新的在前）
    compare_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in compare_files]

def get_all_joints2d_images(base_path=CHECKPOINT_BASE):
    """
    获取当前checkpoint文件夹中的所有2D投影可视化图片，按修改时间排序
    """
    checkpoint_dir = get_latest_checkpoint_dir(base_path)
    if not checkpoint_dir:
        return []
    
    reprojection_vis_dir = checkpoint_dir / "reprojection_vis"
    if not reprojection_vis_dir.exists():
        return []
    
    joints2d_files = list(reprojection_vis_dir.glob("joints2d_gt_comparison*.png"))
    # 按修改时间排序（最新的在前）
    joints2d_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in joints2d_files]

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
        
        .image-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 15px;
            gap: 15px;
        }
        
        .nav-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 1.2em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            flex-shrink: 0;
        }
        
        .nav-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .nav-button:active {
            transform: scale(0.95);
        }
        
        .nav-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .image-counter {
            flex: 1;
            text-align: center;
            color: #2d3748;
            font-weight: 600;
            font-size: 0.95em;
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
                <div class="image-path" id="reprojectionPath" style="background: #f0f4f8; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 0.85em; color: #2d3748; word-break: break-all; border-left: 3px solid #667eea;">
                    加载中...
                </div>
                <img id="reprojectionImage" src="/reprojection_image?index=0&t={{ timestamp }}" alt="Reprojection" />
                <div class="image-nav">
                    <button class="nav-button" onclick="prevReprojection()" id="prevReprojBtn">◀</button>
                    <div class="image-counter">
                        <span id="reprojectionCounter">1 / {{ reprojection_count }}</span>
                    </div>
                    <button class="nav-button" onclick="nextReprojection()" id="nextReprojBtn">▶</button>
                </div>
                {% else %}
                <div class="not-available">Reprojection图像暂不可用</div>
                {% endif %}
            </div>
            
            <div class="image-container">
                <h2>🔍 Compare</h2>
                {% if has_compare %}
                <div class="image-path" id="comparePath" style="background: #f0f4f8; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 0.85em; color: #2d3748; word-break: break-all; border-left: 3px solid #667eea;">
                    加载中...
                </div>
                <img id="compareImage" src="/compare_image?index=0&t={{ timestamp }}" alt="Compare" />
                <div class="image-nav">
                    <button class="nav-button" onclick="prevCompare()" id="prevCompareBtn">◀</button>
                    <div class="image-counter">
                        <span id="compareCounter">1 / {{ compare_count }}</span>
                    </div>
                    <button class="nav-button" onclick="nextCompare()" id="nextCompareBtn">▶</button>
                </div>
                {% else %}
                <div class="not-available">Compare图像暂不可用</div>
                {% endif %}
            </div>
        </div>
        
        <!-- 2D投影可视化 -->
        <div class="image-container">
            <h2>📐 2D Joints Projection</h2>
            {% if has_joints2d %}
            <div class="image-path" id="joints2dPath" style="background: #f0f4f8; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 0.85em; color: #2d3748; word-break: break-all; border-left: 3px solid #667eea;">
                加载中...
            </div>
            <img id="joints2dImage" src="/joints2d_image?index=0&t={{ timestamp }}" alt="2D Joints Projection" />
            <div class="image-nav">
                <button class="nav-button" onclick="prevJoints2d()" id="prevJoints2dBtn">◀</button>
                <div class="image-counter">
                    <span id="joints2dCounter">1 / {{ joints2d_count }}</span>
                </div>
                <button class="nav-button" onclick="nextJoints2d()" id="nextJoints2dBtn">▶</button>
            </div>
            {% else %}
            <div class="not-available">2D投影可视化图像暂不可用</div>
            {% endif %}
        </div>
        
        <div class="controls">
            <button onclick="refreshImages(true)">🔄 立即刷新</button>
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
        let currentReprojectionIndex = 0;
        let currentCompareIndex = 0;
        let currentJoints2dIndex = 0;
        let totalReprojectionImages = {{ reprojection_count }};
        let totalCompareImages = {{ compare_count }};
        let totalJoints2dImages = {{ joints2d_count }};
        
        function refreshImages(resetToLatest = false) {
            const timestamp = new Date().getTime();
            
            // 如果是手动刷新，重置到最新图像
            if (resetToLatest) {
                currentReprojectionIndex = 0;
                currentCompareIndex = 0;
                currentJoints2dIndex = 0;
            }
            
            // 刷新Loss图像
            const lossImg = document.getElementById('lossImage');
            if (lossImg) {
                lossImg.src = '/loss_image?t=' + timestamp;
            }
            
            // 刷新Reprojection图像
            const reprojImg = document.getElementById('reprojectionImage');
            if (reprojImg) {
                reprojImg.src = '/reprojection_image?index=' + currentReprojectionIndex + '&t=' + timestamp;
            }
            
            // 刷新Compare图像
            const compareImg = document.getElementById('compareImage');
            if (compareImg) {
                compareImg.src = '/compare_image?index=' + currentCompareIndex + '&t=' + timestamp;
            }
            
            // 刷新2D投影图像
            const joints2dImg = document.getElementById('joints2dImage');
            if (joints2dImg) {
                joints2dImg.src = '/joints2d_image?index=' + currentJoints2dIndex + '&t=' + timestamp;
            }
            
            // 更新计数器和路径
            if (resetToLatest) {
                updateReprojectionCounter();
                updateCompareCounter();
                updateJoints2dCounter();
            }
            
            refreshCount++;
            document.getElementById('refreshCounter').textContent = '刷新次数: ' + refreshCount;
            document.getElementById('lastUpdate').textContent = '最后更新: ' + new Date().toLocaleTimeString('zh-CN');
        }
        
        function updateReprojectionCounter() {
            document.getElementById('reprojectionCounter').textContent = (currentReprojectionIndex + 1) + ' / ' + totalReprojectionImages;
            document.getElementById('prevReprojBtn').disabled = currentReprojectionIndex === 0;
            document.getElementById('nextReprojBtn').disabled = currentReprojectionIndex === totalReprojectionImages - 1;
            updateReprojectionPath();
        }
        
        function updateCompareCounter() {
            document.getElementById('compareCounter').textContent = (currentCompareIndex + 1) + ' / ' + totalCompareImages;
            document.getElementById('prevCompareBtn').disabled = currentCompareIndex === 0;
            document.getElementById('nextCompareBtn').disabled = currentCompareIndex === totalCompareImages - 1;
            updateComparePath();
        }
        
        function updateJoints2dCounter() {
            document.getElementById('joints2dCounter').textContent = (currentJoints2dIndex + 1) + ' / ' + totalJoints2dImages;
            document.getElementById('prevJoints2dBtn').disabled = currentJoints2dIndex === 0;
            document.getElementById('nextJoints2dBtn').disabled = currentJoints2dIndex === totalJoints2dImages - 1;
            updateJoints2dPath();
        }
        
        function updateReprojectionPath() {
            fetch('/get_image_path?type=reprojection&index=' + currentReprojectionIndex)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('reprojectionPath').textContent = data.path;
                })
                .catch(error => {
                    document.getElementById('reprojectionPath').textContent = '无法获取路径';
                });
        }
        
        function updateComparePath() {
            fetch('/get_image_path?type=compare&index=' + currentCompareIndex)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('comparePath').textContent = data.path;
                })
                .catch(error => {
                    document.getElementById('comparePath').textContent = '无法获取路径';
                });
        }
        
        function updateJoints2dPath() {
            fetch('/get_image_path?type=joints2d&index=' + currentJoints2dIndex)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('joints2dPath').textContent = data.path;
                })
                .catch(error => {
                    document.getElementById('joints2dPath').textContent = '无法获取路径';
                });
        }
        
        function prevReprojection() {
            if (currentReprojectionIndex > 0) {
                currentReprojectionIndex--;
                const timestamp = new Date().getTime();
                document.getElementById('reprojectionImage').src = '/reprojection_image?index=' + currentReprojectionIndex + '&t=' + timestamp;
                updateReprojectionCounter();
            }
        }
        
        function nextReprojection() {
            if (currentReprojectionIndex < totalReprojectionImages - 1) {
                currentReprojectionIndex++;
                const timestamp = new Date().getTime();
                document.getElementById('reprojectionImage').src = '/reprojection_image?index=' + currentReprojectionIndex + '&t=' + timestamp;
                updateReprojectionCounter();
            }
        }
        
        function prevCompare() {
            if (currentCompareIndex > 0) {
                currentCompareIndex--;
                const timestamp = new Date().getTime();
                document.getElementById('compareImage').src = '/compare_image?index=' + currentCompareIndex + '&t=' + timestamp;
                updateCompareCounter();
            }
        }
        
        function nextCompare() {
            if (currentCompareIndex < totalCompareImages - 1) {
                currentCompareIndex++;
                const timestamp = new Date().getTime();
                document.getElementById('compareImage').src = '/compare_image?index=' + currentCompareIndex + '&t=' + timestamp;
                updateCompareCounter();
            }
        }
        
        function prevJoints2d() {
            if (currentJoints2dIndex > 0) {
                currentJoints2dIndex--;
                const timestamp = new Date().getTime();
                document.getElementById('joints2dImage').src = '/joints2d_image?index=' + currentJoints2dIndex + '&t=' + timestamp;
                updateJoints2dCounter();
            }
        }
        
        function nextJoints2d() {
            if (currentJoints2dIndex < totalJoints2dImages - 1) {
                currentJoints2dIndex++;
                const timestamp = new Date().getTime();
                document.getElementById('joints2dImage').src = '/joints2d_image?index=' + currentJoints2dIndex + '&t=' + timestamp;
                updateJoints2dCounter();
            }
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
            updateReprojectionCounter();
            updateCompareCounter();
            updateJoints2dCounter();
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
    all_reprojection = get_all_reprojection_images()
    all_compare = get_all_compare_images()
    all_joints2d = get_all_joints2d_images()
    
    return render_template_string(
        HTML_TEMPLATE, 
        loss_path=latest_loss or LOSS_IMAGE_PATH,
        timestamp=timestamp,
        has_loss=latest_loss is not None,
        has_reprojection=len(all_reprojection) > 0,
        has_compare=len(all_compare) > 0,
        has_joints2d=len(all_joints2d) > 0,
        reprojection_count=len(all_reprojection),
        compare_count=len(all_compare),
        joints2d_count=len(all_joints2d),
        training_time=os.path.dirname(latest_loss).split('/')[-2:] if latest_loss else "N/A"
    )

@app.route('/get_image_path')
def get_image_path():
    """返回指定索引的图像路径"""
    from flask import request, jsonify
    image_type = request.args.get('type', '')
    index = request.args.get('index', 0, type=int)
    
    if image_type == 'reprojection':
        all_images = get_all_reprojection_images()
    elif image_type == 'compare':
        all_images = get_all_compare_images()
    elif image_type == 'joints2d':
        all_images = get_all_joints2d_images()
    else:
        return jsonify({'path': 'Unknown type'}), 400
    
    if all_images and 0 <= index < len(all_images):
        return jsonify({'path': all_images[index]})
    
    return jsonify({'path': 'Not found'}), 404

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
    from flask import request
    index = request.args.get('index', 0, type=int)
    
    all_reprojection = get_all_reprojection_images()
    if all_reprojection and 0 <= index < len(all_reprojection):
        image_path = all_reprojection[index]
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
    
    return "Reprojection image not found", 404

@app.route('/compare_image')
def compare_image():
    """返回compare图像"""
    from flask import request
    index = request.args.get('index', 0, type=int)
    
    all_compare = get_all_compare_images()
    if all_compare and 0 <= index < len(all_compare):
        image_path = all_compare[index]
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
    
    return "Compare image not found", 404

@app.route('/joints2d_image')
def joints2d_image():
    """返回2D投影可视化图像"""
    from flask import request
    index = request.args.get('index', 0, type=int)
    
    all_joints2d = get_all_joints2d_images()
    if all_joints2d and 0 <= index < len(all_joints2d):
        image_path = all_joints2d[index]
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
    
    return "2D Joints image not found", 404

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
    latest_joints2d = get_all_joints2d_images()
    
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
    
    if latest_joints2d:
        print(f"✅ 找到 {len(latest_joints2d)} 个2D投影可视化文件")
        print(f"   最新: {latest_joints2d[0]}")
    else:
        print(f"⚠️  警告: 2D投影可视化文件不存在")
    
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
