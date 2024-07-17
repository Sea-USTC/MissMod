import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=int, default=0)
parser.add_argument('--total_epoch', type=int, default=0)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=50, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='/remote-home/mengxichen/MMANet-CVPR2023-main/output/models')
parser.add_argument('--log_root', type=str, default='/remote-home/mengxichen/MMANet-CVPR2023-main/output/logs')
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')

parser.add_argument('--inplace_new', type=int, default=384, help='para for se layer')
parser.add_argument('--p', default=[0, 0, 0], help='para for modality dropout')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--modal', type=str, default='uni')
parser.add_argument('--begin_epoch', type=int, default=0)

parser.add_argument('--student_name', type=str, default='resnet18_se', help='the backbone for student: resnet18_se')
parser.add_argument('--init_mode', type=str, default='random',
                    help='the way to init the student net: random, rgb, depth, ir')
parser.add_argument('--dataset', type=str, default='cefa')
parser.add_argument("--protocol", type=str, default='race_prot_rdi_4@5')


parser.add_argument('--cuda', type=bool, default=True)


parser.add_argument('--embemdding_dim', type=int, default=512)
parser.add_argument('--miss_modal', type=int, default=0)

parser.add_argument('--network', type=str, default='teacher')
parser.add_argument('--data_root', type=str,
                    default='/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race/')#/remote-home/share/mengxichen/CASIA-SURF/
parser.add_argument('--method', type=str, default='etmc')
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--version', type=int, default=3)
parser.add_argument('--weight_sum', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--weight_kld', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--weight_con', type=float, default=1.0, help='trade-off parameter for kd loss')



args = parser.parse_args()
args.name = args.dataset + "_" + args.method + '_version_' + str(args.version) + '_sum_' + str(
    args.weight_sum)+ '_kld_' + str(args.weight_kld)+ '_con_' + str(args.weight_con)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.drop_mode = 'average'
