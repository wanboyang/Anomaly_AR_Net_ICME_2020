"""
Training module for Anomaly AR Net (ICME 2020) / Anomaly AR Net训练模块 (ICME 2020)
This module contains the main training loop and optimization procedures.
此模块包含主要的训练循环和优化过程。
"""

import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
from losses import KMXMILL_individual, normal_smooth

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    """
    Main training function for Anomaly AR Net / Anomaly AR Net的主要训练函数
    
    Args:
        epochs: Number of training epochs / 训练轮数
        train_loader: DataLoader for training data / 训练数据加载器
        all_test_loader: List of test data loaders / 测试数据加载器列表
        args: Command line arguments / 命令行参数
        model: Neural network model / 神经网络模型
        optimizer: Optimizer for model parameters / 模型参数优化器
        logger: Logger for TensorBoard / TensorBoard日志记录器
        device: Device for computation (CPU/GPU) / 计算设备 (CPU/GPU)
        save_path: Path to save checkpoints and results / 保存检查点和结果的路径
    """
    # Unpack test loaders / 解包测试加载器
    [train2test_loader, test_loader] = all_test_loader
    itr = 0  # Iteration counter / 迭代计数器
    
    # Create result directory if not exists / 如果不存在则创建结果目录
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    
    # Save training arguments to file / 将训练参数保存到文件
    with open(file=os.path.join('./result', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    
    log_statics = {}  # Statistics logging dictionary / 统计日志字典
    
    # Load pretrained model if specified / 如果指定则加载预训练模型
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('model load weights from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')
    
    # Main training loop / 主训练循环
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            itr += 1
            
            # Unpack batch data / 解包批次数据
            [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
            
            # Concatenate anomaly and normal features / 拼接异常和正常特征
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            
            # Calculate sequence lengths / 计算序列长度
            seq_len = torch.sum(torch.max(features.abs(), dim=2)[0] > 0, dim=1).numpy()
            features = features[:, :np.max(seq_len), :]  # Trim to max sequence length / 修剪到最大序列长度

            # Move data to device / 将数据移动到设备
            features = features.float().to(device)
            videolabels = videolabels.float().to(device)
            
            # Forward pass through model / 模型前向传播
            final_features, element_logits = model(features)
            
            # Calculate losses / 计算损失
            weights = args.Lambda.split('_')  # Parse loss weights / 解析损失权重
            m_loss = KMXMILL_individual(element_logits=element_logits,
                                        seq_len=seq_len,
                                        labels=videolabels,
                                        device=device,
                                        loss_type='CE',
                                        args=args)
            n_loss = normal_smooth(element_logits=element_logits,
                                   labels=videolabels,
                                   device=device)

            # Combine losses with weights / 使用权重组合损失
            total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss
            
            # Log losses to TensorBoard / 将损失记录到TensorBoard
            logger.log_value('m_loss', m_loss, itr)
            logger.log_value('n_loss', n_loss, itr)
            
            # Print training progress / 打印训练进度
            if itr % 20 == 0 and not itr == 0:
                print('Iteration:{}, Loss: {}'
                      .format(itr,total_loss.data.cpu().detach().numpy()))
            
            # Backward pass and optimization / 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Save checkpoint and evaluate model / 保存检查点并评估模型
            if itr % args.snapshot == 0 and not itr == 0:
                # Save model checkpoint / 保存模型检查点
                torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'iter_{}'.format(itr) + '.pkl'))
                
                # Test model on test set / 在测试集上测试模型
                test_result_dict = test(test_loader, model, device, args)
                
                # Evaluate performance / 评估性能
                eval_p(itr=itr, dataset=args.dataset_name, predict_dict=test_result_dict, 
                       logger=logger, save_path=save_path, plot=args.plot, args=args)
