import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
from losses import KMXMILL_individual, normal_smooth

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path):
    [train2test_loader, test_loader] = all_test_loader
    itr = 0
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    with open(file=os.path.join('./result', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    log_statics = {}
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('model load weights from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            itr += 1
            [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len = torch.sum(torch.max(features.abs(), dim=2)[0] > 0, dim=1).numpy()
            features = features[:, :np.max(seq_len), :]

            # features = torch.from_numpy(features).float().to(device)
            features = features.float().to(device)
            videolabels = videolabels.float().to(device)
            # final_features, element_logits_linear, element_logits = model(features)
            final_features, element_logits = model(features)
            # if args.model_name == 'model_lstm':
            #     final_features, element_logits = model(features, seq_len)
            # else:
            #     final_features, element_logits = model(features)
            weights = args.Lambda.split('_')
            m_loss = KMXMILL_individual(element_logits=element_logits,
                                        seq_len=seq_len,
                                        labels=videolabels,
                                        device=device,
                                        loss_type='CE',
                                        args=args)
            n_loss = normal_smooth(element_logits=element_logits,
                                   labels=videolabels,
                                   device=device)

            total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss
            logger.log_value('m_loss', m_loss, itr)
            logger.log_value('n_loss', n_loss, itr)
            if itr % 20 == 0 and not itr == 0:
                # print(final_features.shape)
                print('Iteration:{}, Loss: {}'
                      .format(itr,total_loss.data.cpu().detach().numpy()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if itr % args.snapshot == 0 and not itr == 0:
                torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'iter_{}'.format(itr) + '.pkl'))
                test_result_dict = test(test_loader, model, device, args)
                # train_result_dict = test(train2test_loader, model, device, args)
                eval_p(itr=itr, dataset=args.dataset_name, predict_dict=test_result_dict, logger=logger, save_path=save_path, plot=args.plot, args=args)
                # with open(file=os.path.join('./result', save_path, 'predict.pickle'), mode='wb') as f:
                #     pickle.dump(train_result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                # with open(file=os.path.join('./result', save_path, 'log_statics.pickle'), mode='wb') as f:
                #     pickle.dump(log_statics, f, protocol=pickle.HIGHEST_PROTOCOL)

