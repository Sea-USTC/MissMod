import sys

import numpy as np

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi, SURF_MV,SURF_trmlora
from surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_multi
from lib.processing_utils import save_csv
from datasets.surf_txt import SURF, SURF_generate
import torch
import torch.nn as nn
import os


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

    result, _ = calc_accuracy_multi(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)
    return result


if __name__ == '__main__':

    from configuration.config_baseline_multi import args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)
    result_list = []
    result_model = []

    # for k in range(3):
    #     pretrain_dir = "/GPFS/data/mengxichen/MMANet-CVPR2023-main/output/models/multi_full__"+str(k)+"__average.pth"
    #     print(pretrain_dir)
    #     args.gpu = 7
    #     args.modal = 'multi'
    #     args.miss_modal = 0
    #     args.p=[1,1,1]
    #     args.backbone = "resnet18_se"
    #     model = SURF_Baseline(args)
    #     test_para = torch.load(pretrain_dir)
    #     model.load_state_dict(torch.load(pretrain_dir))

    #     result = batch_test(model=model, args=args)



    pretrain_dir = "/DB/public/siyili/lora/output/models/resnet18_se_0__average_acer_best_.pth"
    print(pretrain_dir)
    args.gpu = 0
    args.modal = 'multi'
    args.miss_modal = 0
    args.backbone = "resnet18_se"

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]



    for j in range(len(modality_combination)):
        args.p = modality_combination[j]
        print(args.p)
        model = SURF_Baseline(args)
        model.load_state_dict(torch.load(pretrain_dir))
        result = batch_test(model=model, args=args)
        result_list.append(result)

    result_arr = np.array((result_list))
    print(np.mean(result_arr, axis=0))
    result_model.append(np.mean(result_arr, axis=0))
    result_model = np.array((result_model))
    print(np.mean(result_model, axis=0))
