import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError

# GPU

# 训练参数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
    
parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=False)
parser.add_argument('--total_epoch', type=int, default=0)
parser.add_argument('--miu', type=float, default=0.2)
parser.add_argument('--lam', type=float, default=0.8)
parser.add_argument('--gate01', type= float, default= 0.5)
parser.add_argument('--gate02', type= float, default= 0.5)
parser.add_argument('--only_fused', default=False, dest='only_fused', action='store_true')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', default=False, dest='retrain', action='store_true', help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')
parser.add_argument('--inplace_new', type=int, default=384, help='para for se layer')
parser.add_argument('--p', default=[0, 0, 0], help='para for modality dropout')
parser.add_argument('--modal', type=str, default='multi')
parser.add_argument('--miss_modal', type=int, default=0)
parser.add_argument('--layer_num', type=int, default=4)


parser.add_argument('--network', type=str, default='student')
parser.add_argument('--data_root', type=str,
                    default='/DB/data/mengxichen/MMANet-CVPR2023-main/data/CASIA-SURF')
parser.add_argument('--drop_mode',type=str,default='average')
parser.add_argument('--backbone', type=str, default='resnet_01decomp')
parser.add_argument('--gpu', type=int, default=6)
parser.add_argument('--version', type=int, default=3)

args = parser.parse_args()
args.backbone = args.backbone + '_' + str(args.version) + "_"
args.name = args.backbone + "_" + args.drop_mode
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
