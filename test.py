"""
Testing module for Anomaly AR Net (ICME 2020) / Anomaly AR Net测试模块 (ICME 2020)
This module contains functions for model testing and inference.
此模块包含模型测试和推理的函数。
"""

import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device, args):
    """
    Test the model on test dataset / 在测试数据集上测试模型
    
    Args:
        test_loader: DataLoader for test data / 测试数据加载器
        model: Neural network model / 神经网络模型
        device: Device for computation (CPU/GPU) / 计算设备 (CPU/GPU)
        args: Command line arguments / 命令行参数
    
    Returns:
        result: Dictionary containing predictions for each video / 包含每个视频预测的字典
    """
    result = {}  # Dictionary to store results / 存储结果的字典
    
    # Iterate through test data / 遍历测试数据
    for i, data in enumerate(test_loader):
        feature, data_video_name = data  # Unpack features and video name / 解包特征和视频名称
        feature = feature.to(device)  # Move features to device / 将特征移动到设备
        
        # Perform inference without gradient computation / 在没有梯度计算的情况下执行推理
        with torch.no_grad():
            if args.model_name == 'model_lstm':
                # Handle LSTM model with sequence length / 处理带序列长度的LSTM模型
                _, element_logits = model(feature, seq_len=None, is_training=False)
            else:
                # Handle other models / 处理其他模型
                _, element_logits = model(feature, is_training=False)
        
        # Convert predictions to numpy array / 将预测转换为numpy数组
        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        
        # Store results with video name as key / 以视频名称为键存储结果
        result[data_video_name[0]] = element_logits
    
    return result
