#!/bin/bash
#
# 同步脚本：双向同步本地和远程的dataset_samples目录
# 使用方法：./sync_demo_data.sh [install|start|stop|status|manual-sync|setup-ssh]

set -e

# 配置变量
LOCAL_DIR="/home/zzb/pydata/recons/MEGA/demo_data/input/dataset_samples"
REMOTE_USER="user4"
REMOTE_HOST="125.216.241.151"
REMOTE_DIR="/mnt/Shares/shapenet/Recons/4D-Humans/example_data/examples/dataset_samples"
PID_FILE="/tmp/sync_demo_data.pid"
LOG_FILE="/tmp/sync_demo_data.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%m-%d %H:%M:%S')] ERROR:${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1" >> "$LOG_FILE"
}

# 检查目录是否存在
check_directories() {
    if [ ! -d "$LOCAL_DIR" ]; then
        error "本地目录不存在: $LOCAL_DIR"
        exit 1
    fi

    log "检查远程目录..."
    if ssh -o ConnectTimeout=10 "$REMOTE_USER@$REMOTE_HOST" "if [ -d '$REMOTE_DIR' ]; then echo 'exists'; else echo 'not exists'; fi" 2>/dev/null > /tmp/remote_check.tmp; then
        if grep -q "not exists" /tmp/remote_check.tmp; then
            error "远程目录不存在: $REMOTE_DIR"
            exit 1
        fi
        log "远程目录检查通过"
    else
        error "无法连接到远程服务器，请检查SSH连接"
        exit 1
    fi
}

# 检查SSH连接
check_ssh() {
    log "检查SSH连接..."
    if ssh -o ConnectTimeout=10 -o PasswordAuthentication=no "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH免密登录成功'" 2>/dev/null; then
        log "SSH免密登录正常"
        return 0
    else
        warn "SSH免密登录失败，需要设置SSH免密登录"
        return 1
    fi
}

# 手动同步一次（双向）
manual_sync() {
    log "开始手动双向同步..."

    # 检查SSH连接
    if ! check_ssh; then
        warn "SSH免密登录未设置，同步时可能需要输入密码"
        read -p "继续吗？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    check_directories

    # 同步本地 -> 远程
    log "同步本地 -> 远程..."
    rsync -avz --delete --exclude=".*" --exclude="*.tmp" --exclude="*.swp" \
        -e ssh "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    # 同步远程 -> 本地
    log "同步远程 -> 本地..."
    rsync -avz --delete --exclude=".*" --exclude="*.tmp" --exclude="*.swp" \
        -e ssh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

    log "手动同步完成"
}

# 安装依赖
install_deps() {
    log "安装必要依赖..."

    # 检查并安装unison
    if ! command -v unison &> /dev/null; then
        log "安装unison..."
        sudo apt-get update
        sudo apt-get install -y unison
    else
        log "unison 已安装"
    fi

    # 检查并安装inotify-tools
    if ! command -v inotifywait &> /dev/null; then
        log "安装inotify-tools..."
        sudo apt-get install -y inotify-tools
    else
        log "inotify-tools 已安装"
    fi

    log "依赖安装完成"
}

# 设置SSH免密登录
setup_ssh_key() {
    log "设置SSH免密登录..."

    # 生成SSH密钥（如果不存在）
    if [ ! -f ~/.ssh/id_rsa.pub ]; then
        log "生成SSH密钥..."
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -q
    fi

    log "复制公钥到远程服务器..."
    echo "请手动运行以下命令来设置SSH免密登录:"
    echo "  ssh-copy-id $REMOTE_USER@$REMOTE_HOST"
    echo "或者手动复制公钥:"
    echo "  cat ~/.ssh/id_rsa.pub"
    echo "然后将公钥内容添加到远程服务器的 ~/.ssh/authorized_keys 文件中"

    # 提示用户手动操作
    echo ""
    echo "设置完成后，运行以下命令测试免密登录:"
    echo "  ssh -o PasswordAuthentication=no $REMOTE_USER@$REMOTE_HOST 'echo SSH免密登录成功'"
    echo ""
    echo "如果成功，实时同步将自动工作；否则每次同步需要输入密码"
}

# 实时同步守护进程
start_sync() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        error "同步进程已在运行 (PID: $(cat "$PID_FILE"))"
        exit 1
    fi

    check_directories

    # 检查SSH连接
    if ! check_ssh; then
        warn "SSH免密登录未设置，同步时可能需要输入密码"
        echo "建议先运行: ./sync_demo_data.sh setup-ssh"
        read -p "继续吗？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    log "启动实时同步守护进程..."

    # 启动后台进程
    (
        log "实时同步进程启动"

        while true; do
            # 使用inotifywait监控本地目录变化
            inotifywait -r -e modify,create,delete,move "$LOCAL_DIR" 2>/dev/null | while read path action file; do
                log "检测到本地变化: $action $path$file"

                # 同步到远程
                rsync -avz --delete --exclude=".*" --exclude="*.tmp" --exclude="*.swp" \
                    -e ssh "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" >> "$LOG_FILE" 2>&1

                log "本地 -> 远程同步完成"
            done

            # 如果inotifywait异常退出，等待后重试
            sleep 2
        done
    ) &

    PID=$!
    echo $PID > "$PID_FILE"
    log "实时同步守护进程已启动 (PID: $PID)"
    log "日志输出到: $LOG_FILE"
}

# 停止同步
stop_sync() {
    if [ ! -f "$PID_FILE" ]; then
        error "未找到PID文件，同步进程可能未运行"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        log "停止同步进程 (PID: $PID)..."
        kill $PID
        rm -f "$PID_FILE"
        log "同步进程已停止"
    else
        error "PID $PID 对应的进程不存在"
        rm -f "$PID_FILE"
    fi
}

# 查看状态
status_sync() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        log "同步进程正在运行 (PID: $(cat "$PID_FILE"))"
        echo "最近日志:"
        tail -20 "$LOG_FILE"
    else
        error "同步进程未运行"
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
        fi
    fi
}

# 创建systemd服务文件
create_systemd_service() {
    log "创建systemd服务文件..."

    SERVICE_FILE="/etc/systemd/system/sync-demo-data.service"

    if [ ! -f "$SERVICE_FILE" ]; then
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Synchronize demo_data dataset_samples directory
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(dirname "$(realpath "$0")")
ExecStart=$(realpath "$0") start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
        log "systemd服务文件已创建: $SERVICE_FILE"

        echo ""
        echo "要启用并启动服务，请运行:"
        echo "  sudo systemctl daemon-reload"
        echo "  sudo systemctl enable sync-demo-data"
        echo "  sudo systemctl start sync-demo-data"
        echo "  sudo systemctl status sync-demo-data"
    else
        log "systemd服务文件已存在: $SERVICE_FILE"
    fi
}

# 创建cron任务（定时同步备用方案）
create_cron_job() {
    log "创建cron任务（定时同步）..."

    CRON_JOB="*/5 * * * * $(realpath "$0") manual-sync >> $LOG_FILE 2>&1"

    # 检查是否已有该cron任务
    if crontab -l 2>/dev/null | grep -q "$(realpath "$0")"; then
        log "cron任务已存在"
    else
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        log "cron任务已添加：每5分钟执行一次手动同步"
    fi
}

# 主函数
main() {
    case "$1" in
        install)
            install_deps
            ;;
        setup-ssh)
            setup_ssh_key
            ;;
        start)
            start_sync
            ;;
        stop)
            stop_sync
            ;;
        status)
            status_sync
            ;;
        manual-sync)
            manual_sync
            ;;
        systemd)
            create_systemd_service
            ;;
        cron)
            create_cron_job
            ;;
        *)
            echo "使用方法: $0 [install|setup-ssh|start|stop|status|manual-sync|systemd|cron]"
            echo "  install      安装必要依赖"
            echo "  setup-ssh    设置SSH免密登录"
            echo "  start        启动实时同步守护进程"
            echo "  stop         停止同步进程"
            echo "  status       查看同步状态"
            echo "  manual-sync  执行一次手动双向同步"
            echo "  systemd      创建systemd服务文件（开机自启）"
            echo "  cron         创建cron定时任务（每5分钟同步）"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"