"""
Utility functions for Anomaly AR Net (ICME 2020) / Anomaly AR Net工具函数 (ICME 2020)
This module contains various utility functions for data processing, visualization, and helper operations.
此模块包含用于数据处理、可视化和辅助操作的各种工具函数。
"""

import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Set backend for matplotlib / 设置matplotlib后端
import zipfile
import io
import torch


def random_extract(feat, t_max):
    """
    Randomly extract a contiguous segment from features / 从特征中随机提取连续片段
    
    Args:
        feat: Input features / 输入特征
        t_max: Length of segment to extract / 要提取的片段长度
    
    Returns:
        extracted_feat: Extracted feature segment / 提取的特征片段
        r: Starting index of extraction / 提取的起始索引
    """
    r = np.random.randint(len(feat)-t_max)
    return feat[r:r+t_max], r

def random_extract_step(feat, t_max, step):
    """
    Randomly extract a segment with step sampling / 使用步长采样随机提取片段
    
    Args:
        feat: Input features / 输入特征
        t_max: Length of segment to extract / 要提取的片段长度
        step: Step size for sampling / 采样的步长
    
    Returns:
        extracted_feat: Extracted feature segment / 提取的特征片段
        r: Starting index of extraction / 提取的起始索引
    """
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r


def random_perturb(feat, length):
    """
    Randomly perturb feature sampling with jitter / 使用抖动随机扰动特征采样
    
    Args:
        feat: Input features / 输入特征
        length: Target length for sampling / 采样的目标长度
    
    Returns:
        perturbed_feat: Perturbed feature samples / 扰动的特征样本
        samples: Sampling indices / 采样索引
    """
    samples = np.arange(length) * len(feat) / length
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    # feat = feat[samples]
    return feat[samples.astype('int')], samples.astype('int')



def pad(feat, min_len):
    """
    Pad features to minimum length / 将特征填充到最小长度
    
    Args:
        feat: Input features / 输入特征
        min_len: Minimum length for padding / 填充的最小长度
    
    Returns:
        Padded features / 填充后的特征
    """
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, step):
    """
    Process features by extraction and padding / 通过提取和填充处理特征
    
    Args:
        feat: Input features / 输入特征
        length: Target length / 目标长度
        step: Step size for extraction / 提取的步长
    
    Returns:
        processed_feat: Processed features / 处理后的特征
        r: Extraction index / 提取索引
    """
    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)
            return pad(features, length), r
        else:
            features, r = random_extract(feat, length)
            return features, r
    else:
        return pad(feat, length), 0


def process_feat_sample(feat, length):
    """
    Process features with perturbation sampling / 使用扰动采样处理特征
    
    Args:
        feat: Input features / 输入特征
        length: Target length / 目标长度
    
    Returns:
        processed_feat: Processed features / 处理后的特征
        samples: Sampling indices / 采样索引
    """
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0


def scorebinary(scores=None, threshold=0.5):
    """
    Convert scores to binary values using threshold / 使用阈值将分数转换为二进制值
    
    Args:
        scores: Input scores / 输入分数
        threshold: Binary threshold / 二进制阈值
    
    Returns:
        Binary scores / 二进制分数
    """
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold



def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """
    Fill attention mask inplace for a variable length context / 为可变长度上下文填充注意力掩码
    
    Args:
        mask: Tensor of size (B, N, D) / 形状为(B, N, D)的张量
            Tensor to fill with mask values / 要填充掩码值的张量
        sizes: list[int] / 整数列表
            List giving the size of the context for each item in the batch / 批次中每个项目的上下文大小列表
            Positions beyond each size will be masked / 超出每个大小的位置将被掩码
        v_mask: float / 浮点数
            Value to use for masked positions / 用于掩码位置的值
        v_unmask: float / 浮点数
            Value to use for unmasked positions / 用于非掩码位置的值
    
    Returns:
        mask: Filled with values in {v_mask, v_unmask} / 填充了{v_mask, v_unmask}值的掩码
    """
    mask.fill_(v_unmask)
    n_context = mask.size(2)
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i, :, size:] = v_mask
    return mask


def median(attention_logits, args):
    """
    Apply median-based attention filtering / 应用基于中位数的注意力过滤
    
    This function filters attention logits by keeping only values above the median
    and normalizing the remaining values.
    此函数通过仅保留中位数以上的值并归一化剩余值来过滤注意力logits。
    
    Args:
        attention_logits: Attention logits tensor / 注意力logits张量
        args: Command line arguments / 命令行参数
    
    Returns:
        Filtered and normalized attention logits / 过滤和归一化的注意力logits
    """
    attention_medians = torch.zeros(0).to(args.device)
    # attention_logits_median = torch.zeros(0).to(args.device)
    batch_size = attention_logits.shape[0]
    for i in range(batch_size):
        attention_logit = attention_logits[i][attention_logits[i] > 0].unsqueeze(0)
        attention_medians = torch.cat((attention_medians, attention_logit.median(1, keepdims=True)[0]), dim=0)
    attention_medians = attention_medians.unsqueeze(1)
    attention_logits_mask = attention_logits.clone()
    attention_logits_mask[attention_logits <= attention_medians] = 0
    attention_logits_mask[attention_logits > attention_medians] = 1
    attention_logits = attention_logits * attention_logits_mask
    attention_logits_sum = attention_logits.sum(dim=2, keepdim=True)
    attention_logits = attention_logits / attention_logits_sum
    return attention_logits

#




def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False):
    """
    Generate anomaly detection visualization plots / 生成异常检测可视化图
    
    This function creates plots showing anomaly scores over time with ground truth
    annotations for visual evaluation of model performance.
    此函数创建显示随时间变化的异常分数图，带有真实标注，用于模型性能的视觉评估。
    
    Args:
        predict_dict: Dictionary of predictions for each video / 每个视频的预测字典
        label_dict: Dictionary of ground truth labels / 真实标签字典
        save_path: Path to save plots / 保存图的路径
        itr: Iteration number for naming / 用于命名的迭代编号
        save_root: Root directory for saving / 保存的根目录
        zip: Boolean, whether to save plots to a zip file / 布尔值，是否将图保存到zip文件
    """
    if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
    if zip:
        zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))
        with zipfile.ZipFile(zip_file_name, mode="w") as zf:
            for k, v in predict_dict.items():
                img_name = k + '.jpg'
                predict_np = v.repeat(16)
                label_np = label_dict[k][:len(v.repeat(16))]
                x = np.arange(len(predict_np))
                plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)
                plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
                plt.yticks(np.arange(0, 1.1, step=0.1))
                plt.xlabel('Frames')
                plt.grid(True, linestyle='-.')
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf)
                plt.close()
                zf.writestr(img_name, buf.getvalue())


    else:
        for k, v in predict_dict.items():
            predict_np = v.repeat(16)
            label_np = label_dict[k][:len(v.repeat(16))]
            x = np.arange(len(predict_np))
            plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)
            plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel('Frames')
            plt.ylabel('Anomaly scores')
            plt.grid(True, linestyle='-.')
            plt.legend()
            # plt.show()
            if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:
                os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            else:
                plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))
            plt.close()
