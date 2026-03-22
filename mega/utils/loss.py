import torch
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np
import os

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


def masked_cross_entropy(pred, labels, ignore_index=None, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), "a ... b -> a b ...")[
            :, :n_class
        ]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (
            n_class - 1
        )
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


def visualize_reprojection_2d(gt_2d, pred_2d, vis_path, vis_counter=None, raw_img=None):
    """
    可视化重投影的2D关节点
    Args:
        gt_2d: 真实的2D关节点 [B, N, 3] (x, y, confidence)
        pred_2d: 预测的2D关节点 [B, N, 2]
        vis_path: 保存路径
        vis_counter: 计数器标识
        raw_img: 原始图像 [B, C, H, W]，可选
    """
    counter_str = f"_{vis_counter}" if vis_counter is not None else ""
    save_path = os.path.join(vis_path, f"joints2d_gt_comparison{counter_str}.png")
    visualize_joints_2d(gt_2d[:,:,:2], pred_2d, save_path, raw_img=raw_img)


def reprojection_loss_vis(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    return (l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]).mean()

def reprojection_loss(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v) # [16, 9 , 6890] * [16, 6890, 3]
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    return l1_loss(pred_2d, gt_2d)

def reprojection_loss_conf24(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    return loss.mean()

def reprojection_loss_conf24_vis(gt_2d, pred_v, pred_cam, joints_reg, vis_path, epoch, step_count, raw_img):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    
    visualize_reprojection_2d(
        gt_2d[:, :, :2],
        pred_2d,
        vis_path=vis_path,
        vis_counter=f"epoch{epoch}_{step_count}",
        raw_img=raw_img,
        )
    return loss.mean()

def reprojection_loss_conf(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_3dkpt = pred_3dkpt[:, H36M_TO_J17]
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    return loss.mean()


def reprojection_loss_conf_vis(gt_2d, pred_v, pred_cam, joints_reg, epoch, step_count, raw_img):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_3dkpt = pred_3dkpt[:, H36M_TO_J17]
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    
    visualize_reprojection_2d(
    gt_2d[:, :, :2],
    pred_2d,
    vis_path='./demo_out/joints2d_coco_gt2',
    vis_counter=f"epoch{epoch}_{step_count}",
    raw_img=raw_img,
    )
    return loss.mean()

def visualize_joints_2d(gt_2d, pred_2d=None, save_path="vis.png", img_size=512, raw_img=None):
    """
    可视化真实(GT)和预测(Pred)的2D关节点
    Args:
        gt_2d: 真实的2D关节点 [B, N, 2] 或 [N, 2], 范围 [-1, 1]
        pred_2d: 预测的2D关节点 [B, N, 2] 或 [N, 2], 范围 [-1, 1], 可选
        save_path: 保存路径
        img_size: 图像大小 (仅在raw_img为None时使用)
        raw_img: 原始图像 [B, C, H, W] 或 [C, H, W], 可选
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # --- 辅助函数：统一处理 Tensor 到 Numpy 并取第一个 Batch ---
    def prepare_data(data, is_img=False):
        if data is None: return None
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if is_img:
            return data[0] if data.ndim == 4 else data
        else:
            return data[0] if data.ndim == 3 else data

    gt_np = prepare_data(gt_2d)
    pred_np = prepare_data(pred_2d)
    raw_img_np = prepare_data(raw_img, is_img=True)
    
    # --- 处理底图 ---
    if raw_img_np is not None:
        # 转换通道顺序 [C, H, W] -> [H, W, C]
        if raw_img_np.shape[0] == 3:
            raw_img_np = raw_img_np.transpose(1, 2, 0)
        
        # 归一化处理
        if raw_img_np.max() <= 1.01: # 容忍浮点误差
            img = (raw_img_np * 255).astype(np.uint8)
        else:
            img = raw_img_np.astype(np.uint8).copy()
        
        # 颜色空间转换 (假设输入是 RGB，OpenCV 需要 BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # 创建白色背景
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    h, w = img.shape[:2]

    # --- 坐标转换与绘制函数 ---
    def draw_joints(joints, color, label_prefix=""):
        # 映射 [-1, 1] 到 [0, W/H]
        pts_img = joints.copy()
        pts_img[:, 0] = (joints[:, 0] + 1.0) * 0.5 * w
        pts_img[:, 1] = (joints[:, 1] + 1.0) * 0.5 * h
        
        for i, pt in enumerate(pts_img):
            pos = tuple(pt.astype(int))
            # 绘制实心圆
            cv2.circle(img, pos, 5, color, -1)
            # 绘制黑色边框增加辨识度
            cv2.circle(img, pos, 5, (255, 0, 0), 1)
            # 绘制索引文字
            text_color = (0, 0, 255) if label_prefix == "Pred" else (255, 255, 255)
            cv2.putText(img, str(i), (pos[0] + 5, pos[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # 1. 绘制预测值 (红色 BGR: 0, 0, 255)
    if pred_np is not None:
        draw_joints(pred_np, (0, 0, 255), "Pred")

    # 2. 绘制真实值 (绿色 BGR: 0, 255, 0)
    if gt_np is not None:
        draw_joints(gt_np, (0, 255, 0), "GT")

    # --- 保存结果 ---
    cv2.imwrite(save_path, img)
    status = f"GT (Green)" + (f" & Pred (Red)" if pred_np is not None else "")
    print(f"Saved visualization to {save_path} [{status}]")


def visualize_joints_on_image(raw_img, pred_2d, gt_2d, save_path):
    """
    在原图上绘制预测和真实的2D关节点
    Args:
        raw_img: 原始图像 [B, C, H, W] 或 [C, H, W]，范围在 [0, 1] 或 [0, 255]
        pred_2d: 预测的2D关节点 [B, N, 2] 或 [N, 2]，范围在 [-1, 1]
        gt_2d: 真实的2D关节点 [B, N, 2] 或 [N, 2]，范围在 [-1, 1]
        save_path: 保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换图像为numpy
    if torch.is_tensor(raw_img):
        raw_img = raw_img.detach().cpu().numpy()
    
    # 如果是batch，只取第一个样本
    if raw_img.ndim == 4:
        raw_img = raw_img[0]
    
    # 转换通道顺序 [C, H, W] -> [H, W, C]
    if raw_img.shape[0] == 3:
        raw_img = raw_img.transpose(1, 2, 0)
    
    # 转换值范围到 [0, 255]
    if raw_img.max() <= 1.0:
        raw_img = (raw_img * 255).astype(np.uint8)
    else:
        raw_img = raw_img.astype(np.uint8)
    
    # 确保是RGB格式
    if raw_img.shape[2] == 3:
        img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    else:
        img = raw_img
    
    # 转换关节点为numpy
    if torch.is_tensor(pred_2d):
        pred_2d = pred_2d.detach().cpu().numpy()
    if torch.is_tensor(gt_2d):
        gt_2d = gt_2d.detach().cpu().numpy()
    
    # 如果是batch，只取第一个样本
    if pred_2d.ndim == 3:
        pred_2d = pred_2d[0]
    if gt_2d.ndim == 3:
        gt_2d = gt_2d[0]
    
    h, w = img.shape[:2]
    
    # 将坐标从 [-1, 1] 转换到图像坐标 [0, w] 和 [0, h]
    pred_2d_img = pred_2d.copy()
    pred_2d_img[:, 0] = (pred_2d[:, 0] + 1.0) * 0.5 * w
    pred_2d_img[:, 1] = (pred_2d[:, 1] + 1.0) * 0.5 * h
    
    gt_2d_img = gt_2d.copy()
    gt_2d_img[:, 0] = (gt_2d[:, 0] + 1.0) * 0.5 * w
    gt_2d_img[:, 1] = (gt_2d[:, 1] + 1.0) * 0.5 * h
    
    num_joints = gt_2d.shape[0]
    
    # 绘制真实关节点 (绿色)
    for i, pt in enumerate(gt_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
        cv2.putText(img, f"G{i}", (pt[0] + 8, pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # 绘制预测关节点 (红色)
    for i, pt in enumerate(pred_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 4, (0, 0, 255), -1)
        cv2.putText(img, f"P{i}", (pt[0] + 8, pt[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # 添加图例
    cv2.putText(img, "Green: GT", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Red: Pred", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, f"Joints: {num_joints}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 保存图像
    cv2.imwrite(save_path, img)
    print(f"Saved joints on image to {save_path} (joints={num_joints})")
