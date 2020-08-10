import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import sys
from utils import scorebinary, anomap


def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = './result'
    if dataset == 'shanghaitech':
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)


    with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:
        frame_label_dict = pickle.load(f)
    with open(file=os.path.join(label_dict_path, 'video_label.pickle'), mode='rb') as f:
        video_label_dict = pickle.load(f)
    all_predict_np = np.zeros(0)
    all_label_np = np.zeros(0)
    normal_predict_np = np.zeros(0)
    normal_label_np = np.zeros(0)
    abnormal_predict_np = np.zeros(0)
    abnormal_label_np = np.zeros(0)
    for k, v in predict_dict.items():
        if video_label_dict[k] == [1.]:
            frame_labels = frame_label_dict[k]
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            abnormal_predict_np = np.concatenate((abnormal_predict_np, v.repeat(16)))
            abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels[:len(v.repeat(16))]))
        elif video_label_dict[k] == [0.]:
            frame_labels = frame_label_dict[k]
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            normal_predict_np = np.concatenate((normal_predict_np, v.repeat(16)))
            normal_label_np = np.concatenate((normal_label_np, frame_labels[:len(v.repeat(16))]))

    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
    binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)
    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_label_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count

    abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
    binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
    abnormal_ano_false_alarm = fp / (fp + tn)

    print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    if plot:
        anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip)
    if logger:
        logger.log_value('Test_AUC_all_video', all_auc_score, itr)
        logger.log_value('Test_AUC_abnormal_video', abnormal_auc_score, itr)
        logger.log_value('Test_false_alarm_all_video', all_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_normal_video', normal_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_abnormal_video', abnormal_ano_false_alarm, itr)
    if os.path.exists(os.path.join(save_root,save_path)) == 0:
        os.makedirs(os.path.join(save_root,save_path))
    with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
        f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
        f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
        f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))




