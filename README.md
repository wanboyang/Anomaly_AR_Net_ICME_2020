# Introduction
This repository is for **Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning** (ICME 2020). The original paper can be found [here](https://ieeexplore.ieee.org/document/9102722) or https://arxiv.org/abs/2104.07268. The oral video can be viewed [here](https://www.bilibili.com/video/BV1fT4y1P73i/).

Please cite with the following BibTeX:

```
@inproceedings{anomaly_wan2020icme,
  title={Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning},
  author={Wan, Boyang and Fang, Yuming and Xia, Xue and Mei, Jiajie},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo},
  year={2020}
}
```



## Requirements
* Python 3
* CUDA
* numpy
* tqdm
* [PyTorch](http://pytorch.org/) (1.2)
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by running

```
conda env create -f environment.yaml
```


## Data preparation
1. Download the [i3d features]([link: https://pan.baidu.com/s/1Cn1BDw6EnjlMbBINkbxHSQ password: u4k6])(https://drive.google.com/file/d/193jToyF8F5rv1SCgRiy_zbW230OrVkuT/view?usp=sharing) and change the "dataset_path" to you/path/data

## Visual Feature Extraction
if you want to extract Visual Feature like this project, you can clone this project([https://github.com/wanboyang/anomaly_feature])


## Training

```
python main.py
```
The models and testing results will be created on ./ckpt and ./results respectively

## Acknowledgements
Thanks the contribution of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and awesome PyTorch team.

## Contact
Please contact the first author of the associated paper - Boyang Wan （wanboyangjerry@163.com) for any further queries.
