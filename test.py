import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        with torch.no_grad():
            if args.model_name == 'model_lstm':
                _, element_logits = model(feature, seq_len=None, is_training=False)
            else:
                _, element_logits = model(feature, is_training=False)
        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        # element_logits = F.softmax(element_logits, dim=2)[:, :, 1].cpu().data.numpy()
        # element_logits = element_logits.cpu().data.numpy()
        result[data_video_name[0]] = element_logits
    return result




