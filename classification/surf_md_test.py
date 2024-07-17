import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi, SURF_MV, SURF_MD
from surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_feature_kd import args
import torch
import torch.nn as nn
import os
import numpy as np
import csv

def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/remote-home/share/mengxichen/CASIA-SURF/test"
    txt_dir = root_dir + '/test_private_list.txt'
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


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]


    for i in [0.2]:
        for j in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]:

            result_list = []
            pretrain_dir = "/remote-home/mengxichen/MMANet-CVPR2023-main/output/models/surf_etmc_version_1_sum_1.0_kld_"+ str(i) + "_con_" + str(j) + "_acer_best_.pth"
            log_dir = "/remote-home/mengxichen/MMANet-CVPR2023-main/output/logs/surf_etmc_version_1_sum_1.0_kld_"+ str(i) + "_con_" + str(j) +  ".csv"
            args.gpu = 0
            args.modal = 'uni'
            args.network = 'student'
            args.miss_modal = 0
            args.backbone = "resnet18_se"
            args.inplace_new = 384
            print(pretrain_dir)
            args.buffer = 1

            for j in range(len(modality_combination)):
                args.p = modality_combination[j]
                print(args.p)
                model = SURF_MD(args)
                try:
                    model.load_state_dict(torch.load(pretrain_dir))
                    print(pretrain_dir)
                except Exception as e:
                    sys.exit(1)

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

   