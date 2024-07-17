import sys
import csv
sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi, SURF_MV, SURF_MMANet
from surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_feature_kd import args
import torch
import torch.nn as nn
import os
import numpy as np


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/DB/public/mengxichen/CASIA-SURF/test"
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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
    # test_epoch()

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    result_model = []
    for i in [0.2,0.3,0.4]:
        result_list = []
        pretrain_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-miss/models/surf_mad_auxi_weak-" + str(i) + "_version_8_mad_30.0_mar_0.5_acer_best_.pth"
        log_dir = "/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/output-miss/logs/surf_mad_auxi_weak-"+ str(i) +"_version_8_mad_30.0_mar_0.5.csv"
        args.gpu = 0
        args.modal = 'multi'
        args.network = 'teacher'
        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.inplace_new = 384
        args.transformer = 0
        args.buffer = 1

        for j in range(len(modality_combination)):
            args.p = modality_combination[j]
            print(args.p)

            model = SURF_MMANet(args)
            try:
                test_para = torch.load(pretrain_dir)
                model.load_state_dict(torch.load(pretrain_dir))
            except Exception as e:
                # model.load_state_dict(torch.load(pretrain_dir2))
                # print(pretrain_dir2)
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
