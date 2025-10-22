"""
Model architectures for Anomaly AR Net (ICME 2020) / Anomaly AR Net模型架构 (ICME 2020)
This module contains various neural network models for video anomaly detection.
此模块包含用于视频异常检测的各种神经网络模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from utils import fill_context_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def weights_init(m):
    """
    Initialize weights for neural network layers / 初始化神经网络层的权重
    
    Args:
        m: Neural network module / 神经网络模块
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)  # Xavier uniform initialization / Xavier均匀初始化
        m.bias.data.fill_(0)  # Initialize bias to zero / 将偏置初始化为零


class Model_single(torch.nn.Module):
    """
    Single-layer neural network model for anomaly detection / 用于异常检测的单层神经网络模型
    
    Args:
        n_feature: Number of input features / 输入特征数量
    """
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)  # Fully connected layer / 全连接层
        self.classifier = nn.Linear(n_feature, 1)  # Binary classifier / 二分类器
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation / Sigmoid激活函数
        self.dropout = nn.Dropout(0.7)  # Dropout for regularization / 用于正则化的Dropout
        self.apply(weights_init)  # Initialize weights / 初始化权重

    def forward(self, inputs, is_training=True):
        """
        Forward pass / 前向传播
        
        Args:
            inputs: Input features / 输入特征
            is_training: Whether in training mode / 是否处于训练模式
        
        Returns:
            features: Processed features / 处理后的特征
            predictions: Anomaly predictions / 异常预测
        """
        x = F.relu(self.fc(inputs))  # Apply ReLU activation / 应用ReLU激活
        if is_training:
            x = self.dropout(x)  # Apply dropout during training / 训练时应用dropout
        return x, self.sigmoid(self.classifier(x))  # Return features and predictions / 返回特征和预测


class Filter_Module(nn.Module):
    """
    Filter module for foreground/background separation / 用于前景/背景分离的过滤模块
    
    Args:
        len_feature: Length of input features / 输入特征长度
    """
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                      stride=1, padding=0),
            nn.LeakyReLU()  # LeakyReLU activation / LeakyReLU激活函数
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid activation for probability output / 用于概率输出的Sigmoid激活
        )

    def forward(self, x):
        """
        Forward pass / 前向传播
        
        Args:
            x: Input features with shape (B, T, F) / 输入特征，形状为(B, T, F)
        
        Returns:
            out: Foreground weights with shape (B, T, 1) / 前景权重，形状为(B, T, 1)
        """
        # x: (B, T, F)
        out = x.permute(0, 2, 1)  # Transpose to (B, F, T) / 转置为(B, F, T)
        # out: (B, F, T)
        out = self.conv_1(out)  # Apply first convolution / 应用第一个卷积
        out = self.conv_2(out)  # Apply second convolution / 应用第二个卷积
        out = out.permute(0, 2, 1)  # Transpose back to (B, T, 1) / 转置回(B, T, 1)
        # out: (B, T, 1)
        return out


class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.drop_out(out)
        out = self.conv_3(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out


class BaS_Net(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature)
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.num_segments = num_segments
        self.k = num_segments // 8

    def forward(self, x):
        fore_weights = self.filter_module(x)
        x_supp = fore_weights * x

        cas_base = self.cas_module(x)
        cas_supp = self.cas_module(x_supp)

        score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)
        score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)

        score_base = self.softmax(score_base)
        score_supp = self.softmax(score_supp)

        return score_base, cas_base, score_supp, cas_supp, fore_weights


#
# class Model_single(torch.nn.Module):
#     def __init__(self, n_feature):
#         super(Model_single, self).__init__()
#         self.fc = nn.Linear(n_feature, n_feature)
#         self.classifier = nn.Linear(n_feature, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.7)
#         self.apply(weights_init)
#
#     def forward(self, inputs, is_training=True):
#         x = F.relu(self.fc(inputs))
#         if is_training:
#             x = self.dropout(x)
#         return x, self.classifier(x), self.sigmoid(self.classifier(x))


class Model_mean(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_mean, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)
        self.mean_pooling = nn.AvgPool2d((3, 1))
        # self.weight_conv1 = nn.Conv2d(n_channels, out_channels, kernel_size, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_2 = F.relu(self.conv2(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_3 = F.relu(self.conv3(inputs)).permute(0, 2, 1).unsqueeze(2)
        x = torch.cat((x_1, x_2, x_3), dim=2)
        x = self.mean_pooling(x)
        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))
        x = x.squeeze(2)
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))


class Model_sequence(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_sequence, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = x_3 + x_2
        x = F.relu(self.conv_b2(x))
        x = x_1 + x
        x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x, self.sigmoid(self.classifier(x))

class Model_concatcate(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_concatcate, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        # self.conv_b1 = nn.Conv1d(in_channels=n_feature * 3, out_channels=n_feature, kernel_size=1, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')
        # self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc(x))

        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)

        return x, self.sigmoid(self.classifier(x))

# class Model_attention(nn.Module):
#     def __init__(self, args):
#         super(Model_attention, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.fill_context_mask = fill_context_mask
#         self.args = args
#
#     def forward(self, final_features, element_logits, seq_len, labels):
#         seq_len_list = seq_len.tolist()
#         for i in range(len(element_logits)):
#             if labels[i] == 0:
#                 element_logits[i] = 1 - element_logits[i]
#         element_logits = torch.transpose(element_logits, 2, 1)
#         mask = self.fill_context_mask(mask=element_logits.clone(), sizes=seq_len_list, v_mask=float('-inf'), v_unmask=0)
#         attention_logits = element_logits + mask
#         if self.args.attention_type == 'softmax':
#             attention_logits = F.softmax(attention_logits, dim=2)
#         elif self.args.attention_type == 'sigmoid':
#             attention_logits = self.sigmoid(attention_logits)
#         else:
#             raise ('attention_type is out of option')
#         M = torch.bmm(attention_logits, final_features).squeeze(1)
#
#         return M

class model_lstm(torch.nn.Module):
    def __init__(self, n_feature):
        super(model_lstm, self).__init__()
        self.bidirectlstm = nn.LSTM(
            input_size=n_feature,
            hidden_size=n_feature,
            num_layers=1,
            batch_first=True)
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.7)

    def forward(self, inputs, seq_len, is_training=True):
        if is_training:
            seq_len_list = seq_len.tolist()
            x = pack_padded_sequence(input=inputs, lengths=seq_len_list, batch_first=True, enforce_sorted=False)
            x, _ = self.bidirectlstm(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
            # x = self.dropout(x)
        else:
            x, _ = self.bidirectlstm(inputs)
        return x, self.sigmoid(self.classifier(x))

def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.
    elif model_name == 'model_mean':
        model = Model_mean(feature_size)
    elif model_name == 'model_sequence':
        model = Model_sequence(feature_size)
    elif model_name == 'model_concatcate':
        model = Model_concatcate(feature_size)
    elif model_name == 'model_lstm':
        model = model_lstm(feature_size)
    elif model_name == 'model_bas':
        model = BaS_Net(feature_size)
    else:
        raise ('model_name is out of option')
    return model
