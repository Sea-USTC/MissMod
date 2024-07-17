import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from cefa_baseline_multi_dataloader import cefa_baseline_multi_dataloader
from models.surf_baseline import SURF_ETMCNet
from loss.kd import *
from lib.model_develop import train_ETMC
from configuration.config_etmc import args
from lib.processing_utils import seed_torch


def deeppix_main(args):

    if args.dataset=='surf':

        train_loader = surf_baseline_multi_dataloader(train=True, args=args)
        test_loader = surf_baseline_multi_dataloader(train=False, args=args)
    elif args.dataset=='cefa':
        train_loader = cefa_baseline_multi_dataloader(train=True, args=args, mode='train')
        test_loader = cefa_baseline_multi_dataloader(train=False, args=args, mode='dev')
    else:
        raise Exception('error dataset')

    # seed_torch(2)
    print(args)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.network = 'student'
    args.modal = 'uni'
    student_model = SURF_ETMCNet(args)


    # 如果有GPU
    if torch.cuda.is_available():
        student_model.cuda()
        print("GPU is using")

    # define loss functions


    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')

        optimizer = torch.optim.SGD(student_model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')

        optimizer = torch.optim.Adam(student_model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model}


    train_ETMC(net_dict=nets, optimizer=optimizer,train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    args=args)


if __name__ == '__main__':
    deeppix_main(args)
