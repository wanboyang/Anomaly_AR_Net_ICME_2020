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

## 📁 Project Structure

```
Anomaly_AR_Net_ICME_2020/
├── model.py                    # Neural network model architectures
├── options.py                  # Command line argument parser
├── main.py                     # Main entry point and training setup
├── train.py                    # Training loop and optimization
├── test.py                     # Model testing and evaluation
├── losses.py                   # Custom loss functions
├── utils.py                    # Utility functions and helpers
├── video_dataset_anomaly_balance_uni_sample.py  # Dataset loading and processing
├── environment.yaml            # Conda environment configuration
├── LICENSE                     # MIT License
├── README.md                   # English documentation
└── README_CN.md                # Chinese documentation
```

## 🔧 Code Components

### Core Modules

#### Model Architecture (`model.py`)
- **Multiple model variants** for different temporal modeling approaches
- **Filter Module**: Temporal attention mechanism for foreground/background separation
- **CAS Module**: Class Activation Sequence for temporal localization
- **Multi-scale convolution**: Captures temporal patterns at different resolutions
- **LSTM integration**: Bidirectional LSTM for sequence modeling
- **Comprehensive documentation** with bilingual comments (English/Chinese)

#### Training Pipeline (`train.py`)
- **Weakly supervised learning** with video-level labels
- **Center-guided discriminative learning** for feature separation
- **Multi-instance learning** framework
- **Loss optimization** with various loss functions
- **Detailed training loop** with logging and checkpointing

#### Data Processing (`video_dataset_anomaly_balance_uni_sample.py`)
- **Temporal sequence sampling** with balanced normal/abnormal samples
- **Feature extraction** from pre-computed I3D features
- **Sequence padding** for variable-length videos
- **Multi-dataset support** (ShanghaiTech, UCF-Crime, Avenue)
- **Memory-efficient loading** with optional data dictionary

#### Loss Functions (`losses.py`)
- **Discriminative loss functions** for weakly supervised learning
- **Center-guided learning** to enhance feature discrimination
- **Temporal consistency** for smooth predictions
- **K-Max Multiple Instance Learning (KMXMILL)** loss implementation

#### Utility Functions (`utils.py`)
- **Feature processing** with random extraction and perturbation
- **Attention masking** for variable-length sequences
- **Visualization tools** for anomaly score plotting
- **Data preprocessing** and normalization utilities

#### Configuration Management (`options.py`)
- **Comprehensive argument parsing** for all training/testing parameters
- **Hardware configuration** (GPU selection, memory settings)
- **Dataset and feature specifications**
- **Training hyperparameters** and optimization settings

### Key Features

#### Weak Supervision
- **Video-level labels only** for training
- **Temporal localization** from weak supervision
- **Multi-instance learning** paradigm
- **Balanced sampling** of normal and abnormal videos

#### Temporal Modeling
- **Multi-scale temporal convolution** for different time resolutions
- **Attention mechanisms** for temporal focus
- **Sequence modeling** with LSTM networks
- **Background suppression** for improved anomaly detection

#### Feature Processing
- **I3D feature extraction** for spatio-temporal representation
- **Multi-modal support** (RGB, Flow, Combined features)
- **Feature normalization** and preprocessing
- **Variable-length sequence** handling with padding

#### Code Quality
- **Comprehensive documentation** with bilingual comments
- **Modular architecture** for easy extension
- **Type hints** and clear variable naming
- **Error handling** and validation

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
