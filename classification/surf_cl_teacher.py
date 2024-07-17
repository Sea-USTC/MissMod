import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from cefa_baseline_multi_dataloader import cefa_baseline_multi_dataloader
from models.surf_baseline import SURF_Multi, SURF_CLNet
from loss.kd import *
from lib.model_develop import train_cl_patch_feature
from configuration.config_uncl import args
from lib.processing_utils import seed_torch
import numpy as np
import random

def init_seeds(seed): 
    print('=====> Using fixed random seed: ' + str(seed)) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


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
    student_model = SURF_CLNet(args)
    teacher_model = SURF_CLNet(args)

    # teacher_model.load_state_dict(
    #         torch.load(os.path.join(args.model_root, 'cefa_test-uncl-tea-prefirst-avg5_version_0_kd_1.0_shared_1.0.pt')))


    # 如果有GPU
    if torch.cuda.is_available():
        student_model.cuda()
        teacher_model.cuda()
        print("GPU is using")

    # define loss functions
    criterionKD = CL()
    criterionRe = GCL()
    criterionMSE = Hint()


    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

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
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD, 'criterionRe':criterionRe, 'criterionMSE':criterionMSE}

    train_cl_patch_feature(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    args=args)


if __name__ == '__main__':
    init_seeds(args.seed)
    deeppix_main(args)
