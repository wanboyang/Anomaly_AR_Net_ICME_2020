# Anomaly AR-Net: 基于中心引导判别学习的弱监督视频异常检测

[![论文](https://img.shields.io/badge/论文-ICME_2020-blue)](https://ieeexplore.ieee.org/document/9102722)
[![arXiv](https://img.shields.io/badge/arXiv-2104.07268-red)](https://arxiv.org/abs/2104.07268)
[![许可证](https://img.shields.io/badge/许可证-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **语言**: [English](README.md) | [中文](README_CN.md)

---

## 摘要

本仓库实现了 **Anomaly AR-Net**，这是在 ICME 2020 会议上提出的一种新颖的弱监督视频异常检测框架。我们的方法利用中心引导判别学习，仅使用视频级标签就能有效检测视频序列中的异常。该方法通过结合注意力机制和多尺度特征学习，解决了弱监督设置中时间定位的挑战。

## 🎯 主要特性

- **弱监督学习**: 训练仅需要视频级标签
- **中心引导判别学习**: 增强正常和异常模式之间的特征区分能力
- **多尺度时间建模**: 在不同尺度上捕获时间依赖关系
- **注意力机制**: 聚焦于相关的时间片段
- **多种模型架构**: 支持各种骨干网络进行特征提取

## 🏗️ 模型架构

该框架包含几个关键组件：

### 核心模型:
- **Model_single**: 带dropout的基本线性分类器
- **Model_mean**: 带平均池化的多尺度卷积层
- **Model_sequence**: 带残差连接的序列卷积网络
- **Model_concatcate**: 多尺度特征拼接
- **model_lstm**: 用于时间建模的双向LSTM
- **BaS_Net**: 带注意力机制的背景抑制网络

### 关键组件:
1. **过滤模块**: 为时间片段生成注意力权重
2. **CAS模块**: 用于时间定位的类激活序列
3. **多尺度卷积**: 在不同时间分辨率上捕获特征
4. **注意力机制**: 聚焦于相关视频片段

## 📁 项目结构

```
Anomaly_AR_Net_ICME_2020/
├── model.py                    # 神经网络模型架构
├── options.py                  # 命令行参数解析器
├── main.py                     # 主入口点和训练设置
├── train.py                    # 训练循环和优化
├── test.py                     # 模型测试和评估
├── losses.py                   # 自定义损失函数
├── utils.py                    # 工具函数和辅助函数
├── video_dataset_anomaly_balance_uni_sample.py  # 数据集加载和处理
├── environment.yaml            # Conda环境配置
├── LICENSE                     # MIT许可证
├── README.md                   # 英文文档
└── README_CN.md                # 中文文档
```

## 🔧 代码组件

### 核心模块

#### 模型架构 (`model.py`)
- **多种模型变体** 用于不同的时间建模方法
- **过滤模块**: 用于前景/背景分离的时间注意力机制
- **CAS模块**: 用于时间定位的类激活序列
- **多尺度卷积**: 在不同分辨率下捕获时间模式
- **LSTM集成**: 用于序列建模的双向LSTM
- **完整文档** 包含中英文双语注释

#### 训练流程 (`train.py`)
- **弱监督学习** 使用视频级标签
- **中心引导判别学习** 用于特征分离
- **多示例学习** 框架
- **损失优化** 使用各种损失函数
- **详细训练循环** 包含日志记录和检查点保存

#### 数据处理 (`video_dataset_anomaly_balance_uni_sample.py`)
- **时间序列采样** 包含平衡的正常/异常样本
- **特征提取** 从预计算的I3D特征
- **序列填充** 用于可变长度视频
- **多数据集支持** (ShanghaiTech, UCF-Crime, Avenue)
- **内存高效加载** 可选的数据字典

#### 损失函数 (`losses.py`)
- **判别性损失函数** 用于弱监督学习
- **中心引导学习** 增强特征区分能力
- **时间一致性** 用于平滑预测
- **K-最大多示例学习 (KMXMILL)** 损失实现

#### 工具函数 (`utils.py`)
- **特征处理** 包含随机提取和扰动
- **注意力掩码** 用于可变长度序列
- **可视化工具** 用于异常分数绘图
- **数据预处理** 和归一化工具

#### 配置管理 (`options.py`)
- **全面参数解析** 用于所有训练/测试参数
- **硬件配置** (GPU选择, 内存设置)
- **数据集和特征规范**
- **训练超参数** 和优化设置

### 主要特性

#### 弱监督
- **仅使用视频级标签** 进行训练
- **从弱监督中实现时间定位**
- **多示例学习** 范式
- **平衡采样** 正常和异常视频

#### 时间建模
- **多尺度时间卷积** 用于不同的时间分辨率
- **注意力机制** 用于时间聚焦
- **序列建模** 使用LSTM网络
- **背景抑制** 用于改进异常检测

#### 特征处理
- **I3D特征提取** 用于时空表示
- **多模态支持** (RGB, Flow, 组合特征)
- **特征归一化** 和预处理
- **可变长度序列** 处理与填充

#### 代码质量
- **完整文档** 包含中英文双语注释
- **模块化架构** 便于扩展
- **类型提示** 和清晰的变量命名
- **错误处理** 和验证

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020.git
cd Anomaly_AR_Net_ICME_2020

# 创建环境
conda env create -f environment.yaml
conda activate anomaly_icme
```

### 数据准备

1. 从以下位置下载 I3D 特征：
   - [百度网盘](https://pan.baidu.com/s/1Cn1BDw6EnjlMbBINkbxHSQ) (密码: u4k6)
   - [Google Drive](https://drive.google.com/file/d/193jToyF8F5rv1SCgRiy_zbW230OrVkuT/view?usp=sharing)

2. 解压数据集：
   ```bash
   tar -xvf dataset.tar
   ```

3. 在配置中更新数据集路径

### 视觉特征提取

要提取与此项目类似的视觉特征，请克隆：
```bash
git clone https://github.com/wanboyang/anomaly_feature
```

### 训练

```bash
python main.py
```

模型和测试结果将分别保存在 `./ckpt` 和 `./results` 目录中。

## 📊 性能表现

我们的方法在多个视频异常检测基准上实现了最先进的性能：

- **UCF-Crime**: 在弱监督设置下具有竞争力的性能
- **ShanghaiTech**: 有效的异常定位
- **Avenue**: 对不同异常类型的鲁棒检测

## 📚 引用

如果您发现这项工作对您的研究有用，请引用：

```bibtex
@inproceedings{anomaly_wan2020icme,
  title={Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning},
  author={Wan, Boyang and Fang, Yuming and Xia, Xue and Mei, Jiajie},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo},
  year={2020}
}
```

## 🎥 视频演示

在 [Bilibili](https://www.bilibili.com/video/BV1fT4y1P73i/) 上观看口头报告

## 🤝 致谢

我们感谢 [W-TALC](https://github.com/sujoyp/wtalc-pytorch) 的贡献者和 PyTorch 团队提供的优秀框架。

## 📧 联系方式

如有问题和建议，请联系：
- **万博洋** - wanboyangjerry@163.com
