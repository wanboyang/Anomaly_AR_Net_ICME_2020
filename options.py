"""
Command line argument parser for Anomaly AR Net (ICME 2020) / Anomaly AR Net命令行参数解析器 (ICME 2020)
This module defines all command line arguments for training and testing the anomaly detection model.
此模块定义了训练和测试异常检测模型的所有命令行参数。
"""

import argparse

# Create argument parser / 创建参数解析器
parser = argparse.ArgumentParser(description='AR_Net')

# Hardware and basic training parameters / 硬件和基本训练参数
parser.add_argument('--device', type=int, default=0, help='GPU ID / GPU ID')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001) / 学习率 (默认: 0.0001)')
parser.add_argument('--model_name', default='model_single', help='Model architecture name / 模型架构名称')
parser.add_argument('--loss_type', default='DMIL_C', type=str, help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max / 损失函数类型')
parser.add_argument('--pretrain', type=int, default=0, help='Whether to use pretrained model / 是否使用预训练模型')
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model / 预训练模型检查点路径')

# Testing parameters / 测试参数
parser.add_argument('--testing_path', type=str, default=None, help='time file for test model / 测试模型的时间文件')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model / 测试模型的迭代名称')

# Feature and data parameters / 特征和数据参数
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048) / 特征大小 (默认: 2048)')
parser.add_argument('--batch_size', type=int, default=1, help='number of samples in one iteration / 每次迭代的样本数量')
parser.add_argument('--sample_size', type=int, default=30, help='number of samples in one iteration / 每次迭代的样本数量')
parser.add_argument('--sample_step', type=int, default=1, help='Sample step for temporal sampling / 时间采样的步长')
parser.add_argument('--dataset_name', type=str, default='shanghaitech', help='Dataset name / 数据集名称')
parser.add_argument('--dataset_path', type=str, default='/home/tu-wan/windows4t/dataset', help='path to dir contains anomaly datasets / 包含异常数据集的目录路径')
parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow, combine / 特征模态，选项包含rgb, flow, combine')
parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750) / 训练期间最大序列长度 (默认: 750)')

# Training configuration / 训练配置
parser.add_argument('--Lambda', type=str, default='1_20', help='Lambda parameter for loss function / 损失函数的Lambda参数')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1) / 随机种子 (默认: 1)')
parser.add_argument('--max_epoch', type=int, default=20, help='maximum iteration to train (default: 50000) / 最大训练迭代次数 (默认: 50000)')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D) / 使用的特征类型 I3D 或 C3D (默认: I3D)')
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7 / 特征层 fc6 或 fc7')
parser.add_argument('--k', type=int, default=4, help='value of k for top-k selection / 用于top-k选择的k值')
parser.add_argument('--plot', type=int, default=1, help='whether plot the video anomalous map on testing / 是否在测试时绘制视频异常图')

# Memory and optimization / 内存和优化
parser.add_argument('--larger_mem', type=int, default=0, help='Whether to use larger memory / 是否使用更大的内存')

# Snapshot and label configuration / 快照和标签配置
parser.add_argument('--snapshot', type=int, default=200, help='anomaly sample threshold / 异常样本阈值')
parser.add_argument('--label_type', type=str, default='unary', help='Type of labels (unary, etc.) / 标签类型 (unary等)')

# Commented out parameters for future use / 注释掉的参数供将来使用
# parser.add_argument('--rank', type=int, default=0, help='')
# parser.add_argument('--loss_instance_type', type=str, default='weight', help='mean, weight, weight_center or individual')
# parser.add_argument('--MIL_loss_type', type=str, default='CE', help='CE or MSE')
# parser.add_argument('--u_ratio', type=int, default=10, help='')
# parser.add_argument('--anomaly_smooth', type=int, default=1, help='type of smooth function, all or normal')
# parser.add_argument('--sparise_term', type=int, default=1, help='type of smooth function, all or normal')
# parser.add_argument('--attention_type', type=str, default='softmax', help='type of normalization of attention vector, softmax or sigmoid')
# parser.add_argument('--confidence', type=float, default=0, help='anomaly sample threshold')
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
