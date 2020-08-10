import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import options
# from video_dataset_anomaly_balance_sample import dataset # For anomaly
# from torch.utils.data import DataLoader
# import math
# from utils import fill_context_mask, median


mseloss = torch.nn.MSELoss(reduction='mean')
mseloss_vector = torch.nn.MSELoss(reduction='none')
binary_CE_loss = torch.nn.BCELoss(reduction='mean')
binary_CE_loss_vector = torch.nn.BCELoss(reduction='none')




def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


def hinger_loss(anomaly_score, normal_score):
        return F.relu((1 - anomaly_score + normal_score))


def normal_smooth(element_logits, labels, device):

    """

    :param element_logits:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """
    normal_smooth_loss = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        if labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(element_logits[i]).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)
    return normal_smooth_loss










def KMXMILL_individual(element_logits,
                       seq_len,
                       labels,
                       device,
                       loss_type='CE',
                       args=None):

    """

    :param element_logits:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """
    # [train_video_name, start_index, len_index] = stastics_data
    k = np.ceil(seq_len/args.k).astype('int32')
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        tmp, tmp_index = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        # top_index = np.zeros(len_index[i].numpy())
        # top_predicts = np.zeros(len_index[i].numpy())
        # top_index[tmp_index.cpu().numpy() + start_index[i].numpy()] = 1
        # if train_video_name[i][0] in log_statics:
        #     log_statics[train_video_name[i][0]] = np.concatenate((log_statics[train_video_name[i][0]], np.expand_dims(top_index, axis=0)),axis=0)
        # else:
        #     log_statics[train_video_name[i][0]] = np.expand_dims(top_index, axis=0)
        instance_logits = torch.cat((instance_logits, tmp), dim=0)
        if labels[i] == 1:
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
    if loss_type == 'CE':
        milloss = binary_CE_loss(input=instance_logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=instance_logits, target=real_label)
        return milloss


