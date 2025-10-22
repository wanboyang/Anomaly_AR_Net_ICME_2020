# Anomaly AR-Net: Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning

[![Paper](https://img.shields.io/badge/Paper-ICME_2020-blue)](https://ieeexplore.ieee.org/document/9102722)
[![arXiv](https://img.shields.io/badge/arXiv-2104.07268-red)](https://arxiv.org/abs/2104.07268)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **Language**: [English](README.md) | [中文](README_CN.md)

---

## Abstract

This repository implements **Anomaly AR-Net**, a novel weakly supervised video anomaly detection framework presented at ICME 2020. Our approach leverages center-guided discriminative learning to effectively detect anomalies in video sequences using only video-level labels. The method addresses the challenge of temporal localization in weakly supervised settings by incorporating attention mechanisms and multi-scale feature learning.

## 🎯 Key Features

- **Weakly Supervised Learning**: Requires only video-level labels for training
- **Center-Guided Discriminative Learning**: Enhances feature discrimination between normal and abnormal patterns
- **Multi-scale Temporal Modeling**: Captures temporal dependencies at different scales
- **Attention Mechanisms**: Focuses on relevant temporal segments
- **Multiple Model Architectures**: Supports various backbone networks for feature extraction

## 🏗️ Model Architecture

The framework consists of several key components:

### Core Models:
- **Model_single**: Basic linear classifier with dropout
- **Model_mean**: Multi-scale convolutional layers with average pooling
- **Model_sequence**: Sequential convolutional network with residual connections
- **Model_concatcate**: Multi-scale feature concatenation
- **model_lstm**: Bidirectional LSTM for temporal modeling
- **BaS_Net**: Background Suppression Network with attention mechanisms

### Key Components:
1. **Filter Module**: Generates attention weights for temporal segments
2. **CAS Module**: Class Activation Sequence for temporal localization
3. **Multi-scale Convolution**: Captures features at different temporal resolutions
4. **Attention Mechanisms**: Focuses on relevant video segments

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020.git
cd Anomaly_AR_Net_ICME_2020

# Create environment
conda env create -f environment.yaml
conda activate anomaly_icme
```

### Data Preparation

1. Download I3D features from:
   - [Baidu Netdisk](https://pan.baidu.com/s/1Cn1BDw6EnjlMbBINkbxHSQ) (password: u4k6)
   - [Google Drive](https://drive.google.com/file/d/193jToyF8F5rv1SCgRiy_zbW230OrVkuT/view?usp=sharing)

2. Extract the dataset:
   ```bash
   tar -xvf dataset.tar
   ```

3. Update dataset path in configuration

### Visual Feature Extraction

To extract visual features similar to this project, clone:
```bash
git clone https://github.com/wanboyang/anomaly_feature
```

### Training

```bash
python main.py
```

The models and testing results will be saved in `./ckpt` and `./results` directories respectively.

## 📊 Performance

Our method achieves state-of-the-art performance on multiple video anomaly detection benchmarks:

- **UCF-Crime**: Competitive performance in weakly supervised setting
- **ShanghaiTech**: Effective anomaly localization
- **Avenue**: Robust detection across different anomaly types

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{anomaly_wan2020icme,
  title={Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning},
  author={Wan, Boyang and Fang, Yuming and Xia, Xue and Mei, Jiajie},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo},
  year={2020}
}
```

## 🎥 Video Presentation

Watch the oral presentation on [Bilibili](https://www.bilibili.com/video/BV1fT4y1P73i/)

## 🤝 Acknowledgements

We thank the contributors of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and the PyTorch team for their excellent frameworks.

## 📧 Contact

For questions and suggestions, please contact:
- **Boyang Wan** - wanboyangjerry@163.com
