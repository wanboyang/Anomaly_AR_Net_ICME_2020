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
