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


def reprojection_loss(gt_2d, pred_v, pred_cam, joints_reg, visualize=False, vis_path=None, vis_counter=None, raw_img=None):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    
    # 可视化
    if visualize and vis_path is not None:
        counter_str = f"_{vis_counter}" if vis_counter is not None else ""
        save_path = os.path.join(vis_path, f"joints2d_gt_comparison{counter_str}.png")
        # if raw_img is not None:
        #     visualize_joints_on_image(raw_img, pred_2d, gt_2d, save_path)
        # else:
        visualize_gt_joints_2d(gt_2d, save_path, raw_img=raw_img)
    return l1_loss(pred_2d, gt_2d)


def reprojection_loss_conf(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_3dkpt = pred_3dkpt[:, H36M_TO_J17]
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    return loss.mean()


def visualize_gt_joints_2d(gt_2d, save_path, img_size=512, raw_img=None):
    """
    可视化真实的2D关节点
    Args:
        gt_2d: 真实的2D关节点 [B, N, 2] 或 [N, 2]，范围在 [-1, 1]
        save_path: 保存路径
        img_size: 图像大小（仅在raw_img为None时使用）
        raw_img: 原始图像 [B, C, H, W] 或 [C, H, W]，可选
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为numpy
    if torch.is_tensor(gt_2d):
        gt_2d = gt_2d.detach().cpu().numpy()
    
    # 如果是batch，只取第一个样本
    if gt_2d.ndim == 3:
        gt_2d = gt_2d[0]
    
    # 处理原始图像
    if raw_img is not None:
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
        
        img_size = img.shape[0]
    else:
        # 创建白色背景图像
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 获取关节点数量和图像尺寸
    num_joints = gt_2d.shape[0]
    h, w = img.shape[:2]
    
    # 将坐标从 [-1, 1] 转换到图像坐标
    gt_2d_img = gt_2d.copy()
    gt_2d_img[:, 0] = (gt_2d[:, 0] + 1.0) * 0.5 * w
    gt_2d_img[:, 1] = (gt_2d[:, 1] + 1.0) * 0.5 * h
    
    # 绘制真实关节点 (蓝色圆圈)
    for i, pt in enumerate(gt_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 6, (0, 0, 255), -1)
        cv2.circle(img, pt, 6, (0, 0, 0), 1)  # 黑色边框
        cv2.putText(img, str(i), (pt[0] + 10, pt[1] + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 添加标题和信息
    cv2.putText(img, "Ground Truth Joints", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(img, f"Joints: {num_joints}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite(save_path, img)
    print(f"Saved GT joints visualization to {save_path} (joints={num_joints})")


def visualize_joints_2d(pred_2d, gt_2d, save_path, img_size=512):
    """
    可视化预测的2D关节点和真实的2D关节点
    Args:
        pred_2d: 预测的2D关节点 [B, N, 2] 或 [N, 2]，范围在 [-1, 1]
        gt_2d: 真实的2D关节点 [B, N, 2] 或 [N, 2]，范围在 [-1, 1]
        save_path: 保存路径
        img_size: 图像大小
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为numpy
    if torch.is_tensor(pred_2d):
        pred_2d = pred_2d.detach().cpu().numpy()
    if torch.is_tensor(gt_2d):
        gt_2d = gt_2d.detach().cpu().numpy()
    
    # 如果是batch，只取第一个样本
    if pred_2d.ndim == 3:
        pred_2d = pred_2d[0]
    if gt_2d.ndim == 3:
        gt_2d = gt_2d[0]
    
    # 将坐标从 [-1, 1] 转换到图像坐标 [0, img_size]
    pred_2d_img = (pred_2d + 1.0) * 0.5 * img_size
    gt_2d_img = (gt_2d + 1.0) * 0.5 * img_size
    
    # 创建白色背景图像
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 获取关节点数量
    num_joints = pred_2d.shape[0]
    
    # 定义关节连接关系 (SMPL 24关节点骨架)
    if num_joints == 24:
        # SMPL 24 joints skeleton
        skeleton = [
            [0, 1], [0, 2], [0, 3],  # 骨盆到腿
            [1, 4], [2, 5], [3, 6],  # 大腿
            [4, 7], [5, 8], [6, 9],  # 小腿和脊柱
            [9, 12], [9, 13], [9, 14],  # 脊柱到肩膀和头
            [12, 15], [13, 16], [14, 17],  # 肩膀到肘部
            [15, 18], [16, 19], [17, 20],  # 肘部到手腕
            [18, 21], [19, 22], [20, 23],  # 手腕到手
        ]
    elif num_joints == 17:
        # Human3.6M 17 joints skeleton
        skeleton = [
            [0, 1], [1, 2], [2, 3],  # 右腿
            [0, 4], [4, 5], [5, 6],  # 左腿
            [0, 7], [7, 8], [8, 9], [9, 10],  # 脊柱到头部
            [8, 11], [11, 12], [12, 13],  # 右臂
            [8, 14], [14, 15], [15, 16],  # 左臂
        ]
    else:
        # 如果关节数不匹配，只绘制点，不绘制骨架
        skeleton = []
    
    # 绘制真实关节点的骨架 (绿色)
    for connection in skeleton:
        if connection[0] < num_joints and connection[1] < num_joints:
            pt1 = tuple(gt_2d_img[connection[0]].astype(int))
            pt2 = tuple(gt_2d_img[connection[1]].astype(int))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # 绘制预测关节点的骨架 (红色)
    for connection in skeleton:
        if connection[0] < num_joints and connection[1] < num_joints:
            pt1 = tuple(pred_2d_img[connection[0]].astype(int))
            pt2 = tuple(pred_2d_img[connection[1]].astype(int))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # 绘制真实关节点 (绿色圆圈)
    for i, pt in enumerate(gt_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 5, (0, 255, 0), -1)
        cv2.putText(img, str(i), (pt[0] + 8, pt[1] + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
    
    # 绘制预测关节点 (红色圆圈)
    for i, pt in enumerate(pred_2d_img):
        pt = tuple(pt.astype(int))
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
        cv2.putText(img, str(i), (pt[0] - 15, pt[1] - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
    
    # 添加图例和信息
    cv2.putText(img, "Green: GT", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Red: Pred", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img, f"Joints: {num_joints}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 保存图像
    cv2.imwrite(save_path, img)
    print(f"Saved joints visualization to {save_path} (joints={num_joints})")


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
