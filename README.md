# Introduction
This repository is for **Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning** (ICME 2020). The original paper can be found [here](https://ieeexplore.ieee.org/document/9102722).

Please cite with the following BibTeX:

```
@inproceedings{anomaly_wan2020icme,
  title={Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning},
  author={Wan, Boyang and Fang, Yuming and Xia, Xue and Mei, Jiajie},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo},
  year={2020}
}
```

<p align="center">
  <img src="images/framework.jpg" width="800"/>
</p>


## Requirements
* Python 3
* CUDA
* numpy
* tqdm
* [PyTorch](http://pytorch.org/) (1.2)
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by run

```
conda env create -f environment.yaml
```


## Data preparation
1. Download the [i3d features][link: https://pan.baidu.com/s/1Cn1BDw6EnjlMbBINkbxHSQ password: u4k6] and change the "dataset_path" to you/path/data


## Training

```
python main.py
```
The models and testing results will be created on ./ckpt and ./results respectively

## Acknowledgements
Thanks the contribution of [W-TALC](https://github.com/sujoyp/wtalc-pytorch) and awesome PyTorch team.

## Contact
Please contact the first author of the associated paper - Boyang Wan （wanboyangjerry@163.com) for any further queries.
