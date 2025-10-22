"""
Dataset module for Anomaly AR Net (ICME 2020) / Anomaly AR Net数据集模块 (ICME 2020)
This module contains dataset classes for loading and processing video anomaly detection data.
此模块包含用于加载和处理视频异常检测数据的数据集类。
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    """
    Dataset class for video anomaly detection / 视频异常检测的数据集类
    
    This class handles loading and preprocessing of video features for training and testing.
    此类处理训练和测试的视频特征的加载和预处理。
    """
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        """
        Initialize dataset / 初始化数据集
        
        Args:
            args: Command line arguments / 命令行参数
            train: Boolean indicating training or testing mode / 布尔值，指示训练或测试模式
            trainlist: Custom training list / 自定义训练列表
            testlist: Custom testing list / 自定义测试列表
        """
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        if self.args.larger_mem:
            self.data_dict = self.data_dict_creater()
        if trainlist:
            self.trainlist = self.txt2list(trainlist)
            self.testlist = self.txt2list(testlist)
        else:
            self.trainlist = self.txt2list(
                txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split.txt'))
            self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split.txt'))
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label.pickle'))
        self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict, self.trainlist)
        self.train = train
        self.t_max = args.max_seqlen



    def data_dict_creater(self):
        """
        Create data dictionary for faster loading / 创建数据字典以加快加载速度
        
        Returns:
            data_dict: Dictionary mapping video names to features / 将视频名称映射到特征的字典
        """
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        Generate list from text file / 从文本文件生成列表
        
        Args:
            txtpath: Path of text file / 文本文件路径
        
        Returns:
            filelist: List of file names / 文件名列表
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        """
        Read pickle file / 读取pickle文件
        
        Args:
            file: Path to pickle file / pickle文件路径
        
        Returns:
            Loaded pickle object / 加载的pickle对象
        """
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        """
        Split dataset into normal and anomaly videos / 将数据集拆分为正常和异常视频
        
        Args:
            video_label_dict: Dictionary of video labels / 视频标签字典
            trainlist: List of training videos / 训练视频列表
        
        Returns:
            normal_video_train: List of normal videos / 正常视频列表
            anomaly_video_train: List of anomaly videos / 异常视频列表
        """
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train

        # for k, v in video_label_dict.items():
        #     if v[0] == 1.:
        #         anomaly_video_train.append(k)
        #     else:
        #         normal_video_train.append(k)
        # return normal_video_train, anomaly_video_train

    def __getitem__(self, index):

        if self.args.larger_mem:
            if self.train:
                train_video_name = []
                start_index = []
                anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
                normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
                anomaly_features = torch.zeros(0)
                normaly_features = torch.zeros(0)
                for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                    anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                    normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                    train_video_name += anomaly_data_video_name
                    train_video_name += normaly_data_video_name
                    anomaly_feature = self.data_dict[anomaly_data_video_name]
                    anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                    start_index += r
                    anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                    # shape = (1, seq_len, feature_dim )
                    normaly_feature = self.data_dict[normaly_data_video_name]
                    normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                    start_index += r
                    normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                    anomaly_features = torch.cat((anomaly_features, anomaly_feature),
                                                 dim=0)  # combine anomaly_feature of different a_i
                    normaly_features = torch.cat((normaly_features, normaly_feature),
                                                 dim=0)  # combine normaly_feature of different n_i
                # normaly_label = torch.zeros((self.args.sample_size, 1))
                # anomaly_label = torch.ones((self.args.sample_size, 1))
                if args.label_type == 'binary':
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)
                else:
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.zeros((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)

                return [anomaly_features, normaly_features], [anomaly_label, normaly_label], [train_video_name,start_index]
            else:
                data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
                self.feature = self.data_dict[data_video_name]
                return self.feature, data_video_name

        else:
            if self.train:
                anomaly_train_video_name = []
                normaly_train_video_name = []
                anomaly_start_index = []
                anomaly_len_index = []
                normaly_start_index = []
                normaly_len_index = []
                anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
                normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
                anomaly_features = torch.zeros(0)
                normaly_features = torch.zeros(0)
                for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                    anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                    normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                    anomaly_train_video_name.append(anomaly_data_video_name)
                    normaly_train_video_name.append(normaly_data_video_name)
                    anomaly_feature = np.load(
                        file=os.path.join(self.feature_path, anomaly_data_video_name, 'feature.npy'))
                    anomaly_len_index.append(anomaly_feature.shape[0])
                    anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                    anomaly_start_index.append(r)
                    anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                    normaly_feature = np.load(
                        file=os.path.join(self.feature_path, normaly_data_video_name, 'feature.npy'))
                    normaly_len_index.append(normaly_feature.shape[0])
                    normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                    normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                    normaly_start_index.append(r)
                    anomaly_features = torch.cat((anomaly_features, anomaly_feature),
                                                 dim=0)  # combine anomaly_feature of different a_i
                    normaly_features = torch.cat((normaly_features, normaly_feature),
                                                 dim=0)  # combine normaly_feature of different n_i
                if self.args.label_type == 'binary':
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)
                elif self.args.label_type == 'unary':
                    normaly_label = torch.zeros((self.args.sample_size, 1))
                    anomaly_label = torch.ones((self.args.sample_size, 1))
                else:
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.zeros((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)

                train_video_name = anomaly_train_video_name + normaly_train_video_name
                start_index = anomaly_start_index + normaly_start_index
                len_index = anomaly_len_index + normaly_len_index

                return [anomaly_features, normaly_features], [anomaly_label, normaly_label], [train_video_name, start_index, len_index]
            else:
                data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
                self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
                return self.feature, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)


class dataset_train2test(Dataset):
    """
    Dataset class for testing on training data / 用于在训练数据上测试的数据集类
    
    This class provides access to training data for evaluation purposes.
    此类提供对训练数据的访问以进行评估目的。
    """
    def __init__(self, args, trainlist=None):
        """
        Initialize dataset for training data testing / 初始化用于训练数据测试的数据集
        
        Args:
            args: Command line arguments / 命令行参数
            trainlist: Custom training list / 自定义训练列表
        """
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        if self.args.larger_mem:
            self.data_dict = self.data_dict_creater()
        if trainlist:
            self.trainlist = self.txt2list(trainlist)
        else:
            self.trainlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split.txt'))
    def data_dict_creater(self):
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train


    def __getitem__(self, index):
            data_video_name = self.trainlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
            return self.feature, data_video_name

    def __len__(self):
        return len(self.trainlist)



if __name__ == "__main__":
    args = options.parser.parse_args()
    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=5, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=5, shuffle=False)
    for epoch in range(1):
        for i, data in enumerate(test_loader):
            features, _ = data
            print(features.shape)
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            [anomaly_features, normaly_features], [anomaly_label, normaly_label] = data
            print(anomaly_features.squeeze(0).shape)
            print(normaly_label.squeeze(0).shape)
            #
            # # 将这些数据转换成Variable类型
            # inputs = Variable(imagedata)

            # 接下来就是跑模型的环节了，我们这里使用print来代替
            # print("epoch：", epoch, "的第", i, "个inputs", filedata.shape, real_len, fileinputs)
        # for i, data in enumerate(test_loader):
        #     # 将数据从 train_loader 中读出来,一次读取的样本数是32个        #     filedata, fileinputs= data
        #     #
        #     # # 将这些数据转换成Variable类型
        #     # inputs = Variable(imagedata)
        #
        #     # 接下来就是跑模型的环节了，我们这里使用print来代替
        #     print("epoch：", epoch, "的第" , i+1, "个inputs", filedata.shape, fileinputs)
