import sys

import numpy as np

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_UNCLLateNet
from surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from lib.processing_utils import save_csv
from datasets.surf_txt import SURF, SURF_generate
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

    root_dir = "/DB/public/mengxichen/CASIA-SURF/test"
    txt_dir = root_dir + '/test_private_list.txt'
    # root_dir = "/remote-home/share/mengxichen/CASIA-SURF/valid"
    # txt_dir = "/remote-home/share/mengxichen/CASIA-SURF" + '/val_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal, times=1)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4)

    result, _ = calc_accuracy_kd_patch_feature(model=model, loader=test_loader, args=args, verbose=True, hter=True)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
    # test_epoch()
    
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    # for i in [0.05,0.2,0.3,0.4,0.5]:
    for i in [2.0]:#,2.8,3,3.2,3.4]:
    # for i in [90,100,120,130]:
    # for i in [10,11,12]:

            result_list = []
            pretrain_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-1/models/surf_cat-2fc("+str(0.5)+ ")-2_version_"+str(10)+"_kd_"+ str(i) + "_shared_" + str(i) + "_acer_best_.pth" #".pt"#
            log_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-1/logs/surf_cat-2fc("+str(0.5)+")-2_version_"+str(10)+"_kd_"+ str(i) + "_shared_" + str(i) + ".csv"
            # pretrain_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-miss/models/surf_uncl_train_missing-"+str(i)+"_version_10_kd_1.8_shared_1.8_acer_best_.pth" #.pt"#
            # log_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-miss/logs/surf_uncl_train_missing-"+str(i)+"_version_10_kd_1.8_shared_1.8.csv"
            args.gpu = 0
            args.modal = 'uni'
            args.network = 'student'
            args.miss_modal = 0
            args.backbone = "resnet18_se"
            args.inplace_new = 384
            print(pretrain_dir)
            try:
                for j in range(len(modality_combination)):
                    args.p = modality_combination[j]
                    print(args.p)
                    model = SURF_UNCLLateNet(args)
                    model.load_state_dict(torch.load(pretrain_dir))# ['model_state']
                    print(pretrain_dir)
                    model.avg = False
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
            except Exception as e:
                        # model.load_state_dict(torch.load(pretrain_dir2))
                        # print(pretrain_dir2)
                continue




    # test_single()
