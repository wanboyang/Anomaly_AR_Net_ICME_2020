from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset, dataset_train2test  # For anomaly
from torch.utils.data import DataLoader
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import glob


if __name__ == '__main__':
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    time = datetime.datetime.now()

    save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, 'k_{}'.format(args.k), '_Lambda_{}'.format(args.Lambda), args.feature_modal, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))

    # if args.feature_pretrain_model == 'c3d':
    #     save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.feature_layer, args.dataset_name,'k_{}'.format(args.k),
    #                              args.loss_type + '_Lambda_{}'.format(args.Lambda) + '_rank_{}'.format(args.rank) +
    #                              '_loss_instance_type_{}'.format(args.loss_instance_type) + '_confidence_{}'.format(args.confidence),
    #                              args.feature_modal,'{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour, time.minute, time.second))
    # else:
    #     save_path = os.path.join(args.model_name, args.feature_pretrain_model,args.dataset_name,'k_{}'.format(args.k),
    #                              args.loss_type + '_Lambda_{}'.format(args.Lambda) + '_rank_{}'.format(args.rank) +
    #                              '_loss_instance_type_{}'.format(args.loss_instance_type) + '_confidence_{}'.format(args.confidence),
    #                              args.feature_modal,'{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour, time.minute, time.second))


    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                              num_workers=1, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    train2test_dataset = dataset_train2test(args=args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)
    train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
                                   num_workers=2, shuffle=False)
    all_test_loader = [train2test_loader, test_loader]

    # if args.dataset_name == 'Avenue':
    #     train_test_split_path = os.path.join(args.dataset_path, args.dataset_name, 'Avenuetxt')
    #     train_lists = glob.glob(os.path.join(train_test_split_path, '*train*.txt'))
    #     test_lists = glob.glob(os.path.join(train_test_split_path, '*test*.txt'))
    # elif args.dataset_name == 'UCSDPed2':
    #     train_test_split_path = os.path.join(args.dataset_path, args.dataset_name, 'Ped2txt')
    #     train_lists = glob.glob(os.path.join(train_test_split_path, '*train*.txt'))
    #     test_lists = glob.glob(os.path.join(train_test_split_path, '*test*.txt'))
    # else:
    #     train_lists = None
    #     test_lists = None
    # if train_lists:
    #     trainiters = 0
    #     for train_list, test_list in zip(train_lists, test_lists):
    #         train_dataset = dataset(args=args, train=True, trainlist=train_list, testlist=test_list)
    #         train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
    #                                   num_workers=2, shuffle=True)
    #         train2test_dataset = dataset_train2test(args=args, trainlist=train_list)
    #         test_dataset = dataset(args=args, train=False, trainlist=train_list, testlist=test_list)
    #         train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
    #                                  num_workers=2, shuffle=False)
    #         test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
    #                                  num_workers=2, shuffle=False)
    #         all_test_loader = [train2test_loader, test_loader]
    #         save_path_t = save_path + '/split_{}/'.format(trainiters)
    #         if not os.path.exists('./ckpt/' +save_path_t):
    #             os.makedirs('./ckpt/' + save_path_t)
    #         if not os.path.exists('./logs/' + save_path_t):
    #             os.makedirs('./logs/' + save_path_t)
    #         logger = Logger('./logs/' + save_path_t)
    #         train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model,
    #               optimizer=optimizer, logger=logger, device=device, save_path=save_path_t)
    #         trainiters += 1
    #         model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    #         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    #
    # else:
    #     train_dataset = dataset(args=args, train=True)
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
    #                               num_workers=1, shuffle=True)
    #     test_dataset = dataset(args=args, train=False)
    #     train2test_dataset = dataset_train2test(args=args)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
    #                              num_workers=2, shuffle=False)
    #     train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
    #                                    num_workers=2, shuffle=False)
    #     all_test_loader = [train2test_loader, test_loader]
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    logger = Logger('./logs/'+ save_path)
    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model, optimizer=optimizer, logger=logger, device=device, save_path=save_path)
