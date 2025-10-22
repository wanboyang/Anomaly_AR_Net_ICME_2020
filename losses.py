"""
Loss functions for Anomaly AR Net (ICME 2020) / Anomaly AR Net损失函数 (ICME 2020)
This module contains custom loss functions for weakly supervised anomaly detection.
此模块包含用于弱监督异常检测的自定义损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import options
# from video_dataset_anomaly_balance_sample import dataset # For anomaly
# from torch.utils.data import DataLoader
# import math
# from utils import fill_context_mask, median

# Predefined loss functions / 预定义的损失函数
mseloss = torch.nn.MSELoss(reduction='mean')  # Mean squared error loss / 均方误差损失
mseloss_vector = torch.nn.MSELoss(reduction='none')  # MSE loss without reduction / 无缩减的MSE损失
binary_CE_loss = torch.nn.BCELoss(reduction='mean')  # Binary cross entropy loss / 二元交叉熵损失
binary_CE_loss_vector = torch.nn.BCELoss(reduction='none')  # BCE loss without reduction / 无缩减的BCE损失




def cross_entropy(logits, target, size_average=True):
    """
    Custom cross entropy loss function / 自定义交叉熵损失函数
    
    Args:
        logits: Model predictions / 模型预测
        target: Ground truth labels / 真实标签
        size_average: Whether to average the loss / 是否对损失求平均
    
    Returns:
        Cross entropy loss / 交叉熵损失
    """
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


def hinger_loss(anomaly_score, normal_score):
    """
    Hinge loss for anomaly detection / 用于异常检测的铰链损失
    
    Args:
        anomaly_score: Anomaly prediction scores / 异常预测分数
        normal_score: Normal prediction scores / 正常预测分数
    
    Returns:
        Hinge loss value / 铰链损失值
    """
    return F.relu((1 - anomaly_score + normal_score))


def normal_smooth(element_logits, labels, device):
    """
    Smoothness loss for normal videos / 正常视频的平滑损失
    
    This loss encourages temporal consistency in normal videos by minimizing
    the variance of predictions across frames.
    该损失通过最小化跨帧预测的方差来鼓励正常视频的时间一致性。
    
    Args:
        element_logits: Frame-level predictions / 帧级预测
        labels: Video-level labels / 视频级标签
        device: Computation device / 计算设备
    
    Returns:
        Smoothness loss for normal videos / 正常视频的平滑损失
    """
    normal_smooth_loss = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    # 因为批次的实际大小可能不等于批次大小（对于epoch中的最后一个批次）
    for i in range(real_size):
        if labels[i] == 0:  # Normal video / 正常视频
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(element_logits[i]).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)
    return normal_smooth_loss










def KMXMILL_individual(element_logits,
                       seq_len,
                       labels,
                       device,
                       loss_type='CE',
                       args=None):
    """
    K-Max Multiple Instance Learning (KMXMILL) loss / K-最大多示例学习损失
    
    This loss function implements the K-Max MIL approach for weakly supervised
    anomaly detection. It selects the top-k predictions from each video and
    computes the loss based on these selected instances.
    该损失函数实现了用于弱监督异常检测的K-最大MIL方法。它从每个视频中选择前k个预测，
    并基于这些选定的实例计算损失。
    
    Args:
        element_logits: Frame-level predictions / 帧级预测
        seq_len: Sequence lengths for each video / 每个视频的序列长度
        labels: Video-level labels / 视频级标签
        device: Computation device / 计算设备
        loss_type: Type of loss ('CE' or 'MSE') / 损失类型 ('CE' 或 'MSE')
        args: Command line arguments / 命令行参数
    
    Returns:
        KMXMILL loss value / KMXMILL损失值
    """
    # [train_video_name, start_index, len_index] = stastics_data
    # Calculate k for each video based on sequence length / 基于序列长度为每个视频计算k
    k = np.ceil(seq_len/args.k).astype('int32')
    instance_logits = torch.zeros(0).to(device)  # Store top-k predictions / 存储前k个预测
    real_label = torch.zeros(0).to(device)  # Store corresponding labels / 存储对应的标签
    real_size = int(element_logits.shape[0])
    
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    # 因为批次的实际大小可能不等于批次大小（对于epoch中的最后一个批次）
    for i in range(real_size):
        # Select top-k predictions for current video / 为当前视频选择前k个预测
        tmp, tmp_index = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        
        # Store selected predictions and corresponding labels / 存储选定的预测和对应的标签
        instance_logits = torch.cat((instance_logits, tmp), dim=0)
        if labels[i] == 1:  # Anomaly video / 异常视频
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:  # Normal video / 正常视频
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
    
    # Compute final loss based on loss type / 基于损失类型计算最终损失
    if loss_type == 'CE':
        milloss = binary_CE_loss(input=instance_logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=instance_logits, target=real_label)
        return milloss
