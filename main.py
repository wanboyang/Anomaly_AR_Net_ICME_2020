"""
Main entry point for Anomaly AR Net (ICME 2020) / Anomaly AR Net主入口 (ICME 2020)
This script sets up the training environment, loads data, and starts the training process.
此脚本设置训练环境，加载数据，并启动训练过程。
"""

from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset, dataset_train2test  # For anomaly
from torch.utils.data import DataLoader
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import glob


if __name__ == '__main__':
    """
    Main function for training Anomaly AR Net / 训练Anomaly AR Net的主函数
    """
    # Parse command line arguments / 解析命令行参数
    args = options.parser.parse_args()
    
    # Set random seed and device / 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    time = datetime.datetime.now()

    # Create save path with timestamp / 创建带时间戳的保存路径
    save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, 
                            'k_{}'.format(args.k), '_Lambda_{}'.format(args.Lambda), 
                            args.feature_modal, 
                            '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, 
                            time.day, time.hour, time.minute, time.second))

    # Initialize model and optimizer / 初始化模型和优化器
    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    # Load pretrained model if specified / 如果指定则加载预训练模型
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    # Create datasets and data loaders / 创建数据集和数据加载器
    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                              num_workers=1, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    train2test_dataset = dataset_train2test(args=args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)
    train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
                                   num_workers=2, shuffle=False)
    all_test_loader = [train2test_loader, test_loader]

    # Create directories for checkpoints and logs / 创建检查点和日志目录
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    
    # Initialize logger for TensorBoard / 初始化TensorBoard日志记录器
    logger = Logger('./logs/'+ save_path)
    
    # Start training process / 启动训练过程
    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, 
          args=args, model=model, optimizer=optimizer, logger=logger, device=device, save_path=save_path)
