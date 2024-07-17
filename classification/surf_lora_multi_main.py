import sys

sys.path.append('..')
from models.surf_baseline import SURF_trmlora
from surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from configuration.config_lora_multi import args
from lib.model_develop import train_lora_multi
from lib.processing_utils import get_file_list
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import datetime
import random
import os


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):
    train_loader = surf_baseline_multi_dataloader(train=True, args=args)
    test_loader = surf_baseline_multi_dataloader(train=False, args=args)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.epoch = 0
    print(type(args.p))
    try:
     args.p = eval(args.p)
    except Exception as e:
        print(1)
    # print(args.p)
    # print(type(args.p))
    model = SURF_trmlora(args)
    # model = SURF_UNCLBaseline(args)
    if torch.cuda.is_available():
        model.cuda()
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    # args.retrain = False
    train_lora_multi(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
    deeppix_main(args=args)
