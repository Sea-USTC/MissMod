import sys

import numpy as np

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_RAMLNet
from cefa_baseline_multi_dataloader import cefa_multi_transforms_test, cefa_multi_transforms_train
from lib.model_develop import calc_accuracy_multi
from lib.processing_utils import save_csv
from datasets.cefa_multi_protocol import CEFA_Multi
from configuration.cefa_baseline_multi import args
import torch
import torch.nn as nn
import os
import csv

def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    args.data_root = "/remote-home/share/mengxichen/CASIA-CeFA/CeFA-Race/CeFA-Race"
    cefa_dataset = CEFA_Multi(args=args, mode='test', miss_modal=args.miss_modal, protocol=args.protocol,
                              transform=cefa_multi_transforms_test)
    cefa_data_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)

    result,_ = calc_accuracy_multi(model=model, loader=cefa_data_loader, verbose=True, hter=True)
    print(result)
    return result


def test_epoch():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    result_model = []
    result_epoch = []
    result_multi_epoch = []
    for i in range(3):
        i = i
        for k in range(99):
            result_list = []
            pretrain_dir = "../output/models/multi_baseline_" + str(3) + "__average_epoch_" + str(k) + ".pth"
            args.gpu = 1
            args.modal = 'multi'
            args.miss_modal = 0
            args.backbone = "resnet18_se"
            args.inplace_new = 384
            print(pretrain_dir)

            for j in range(len(modality_combination)):
                args.p = modality_combination[j]
                print(args.p)
                model = SURF_Baseline(args)
                model.load_state_dict(torch.load(pretrain_dir))

                result = batch_test(model=model, args=args)
                result_list.append(result)
                result_epoch.append(result[3])
            result_multi_epoch.append(result_epoch)
            result_arr = np.array((result_list))
            result_mean = np.mean(result_arr, axis=0)
            print(result_mean)
            result_model.append(result_mean)
        for i in range(len(result_multi_epoch)):
            save_csv("../output/multi_baseline_average_lr0.01_epoch.csv", result_multi_epoch[i])


def test_single():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # test_epoch()
    result_list = []
    args.gpu = 1
    args.modal = 'multi'
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.inplace_new = 384

    modality_combination = [[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    for j in range(len(modality_combination)):

        pretrain_dir = "../output/models/cefa_baseline_single_" + str(j) + "__average.pth"

        args.p = modality_combination[j]
        print(args.p)
        model = SURF_Baseline(args)
        test_para = torch.load(pretrain_dir)
        model.load_state_dict(torch.load(pretrain_dir))

        result = batch_test(model=model, args=args)
        result_list.append(result)

        print(result)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
    # test_epoch()
    
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    for i in [1.6,1.8]:
        for j in [0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:

            result_list = []
            pretrain_dir = "/remote-home/mengxichen/MMANet-CVPR2023-main/output/models/cefa_raml_version_1_uni_"+ str(i) + "_inf_" + str(j)+ ".pth"
            log_dir = "/remote-home/mengxichen/MMANet-CVPR2023-main/output/logs/cefa_raml_version_1_uni_"+ str(i) + "_inf_" + str(j)+ ".csv"
            args.gpu = 0
            args.modal = 'uni'
            args.network = 'student'
            args.miss_modal = 0
            args.backbone = "resnet18_se"
            args.inplace_new = 384
            print(pretrain_dir)

            for j in range(len(modality_combination)):
                args.p = modality_combination[j]
                print(args.p)
                model = SURF_RAMLNet(args)
                test_para = torch.load(pretrain_dir)
                model.load_state_dict(torch.load(pretrain_dir))

                result = batch_test(model=model, args=args)
                with open(log_dir, 'a+', newline='') as f:
                    my_writer = csv.writer(f)
                    my_writer.writerow(result)
                result_list.append(result)

            result_arr = np.array((result_list))
            result_mean = np.mean(result_arr, axis=0)
            print(result_mean)
            with open(log_dir, 'a+', newline='') as f:
            # 训练结果
                my_writer = csv.writer(f)
                my_writer.writerow(result_mean)


    # test_single()
