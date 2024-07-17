'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
import time

import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy
from loss.mmd_loss import MMD_loss
from loss.kd import UNCL, GCL
import datetime
from datasets.surf_txt import SURF, SURF_generate
from surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.processing_utils import get_dataself_hist
import math



def calc_accuracy_multi(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_rgb, img_depth, img_ir)
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            APCER = living_wrong / (living_wrong + living_right)
            NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)
            ACER = (APCER + NPCER) / 2

            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong), labels_predicted
            return [accuracy, 0, 0, 0], labels_predicted

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, APCER, NPCER, ACER], labels_predicted
    else:
        return [accuracy], labels_predicted


def calc_accuracy_kd_patch_feature(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    # use_cuda = True
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    criterionKD = UNCL()
    criterionRe = GCL(0.1)
    kd_loss = 0

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if use_cuda:
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():

            # outputs_batch = model(img_rgb, img_depth, img_ir)
            # if isinstance(outputs_batch, tuple):
            #     # outputs_batch = outputs_batch[0]
            outputs_batch, fusion_mu, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, _, p = model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch

            index = p.squeeze().sum(-1) == 3
              
            kd_shared_loss = criterionKD(rgb_mu_full[index], fusion_mu[index], rgb_logvar_full[index], target[index], False) + \
                             criterionKD(ir_mu_full[index], fusion_mu[index], ir_logvar_full[index], target[index], False) + \
                             criterionKD(depth_mu_full[index], fusion_mu[index], depth_logvar_full[index], target[index], False) 
                             
                                #   criterionKD(fusion_mu[multi_index], teacher_layer.detach(), fusion_logvar[multi_index], target, False, multi_index) 
            # if torch.isnan(kd_shared_loss):
            #     print("Loss is nan")
            #     kd_shared_loss = torch.zeros(1).float().cuda()

            kd_loss = kd_loss + kd_shared_loss.item()
 
                              
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)
    kd_loss = kd_loss / len(loader)

    print(kd_loss)
    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            APCER = living_wrong / (living_wrong + living_right)
            NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)
            ACER = (APCER + NPCER) / 2

            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong), labels_predicted
            return [accuracy, 0, 0, 0], labels_predicted

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, APCER, NPCER, ACER], labels_predicted, kd_loss
    else:
        return [accuracy], labels_predicted, kd_loss


# def calc_accuracy_kd_patch_feature(model, loader, args, verbose=False, hter=False):
#     """
#     :param model: model network
#     :param loader: torch.utils.data.DataLoader
#     :param verbose: show progress bar, bool
#     :return accuracy, float
#     """
#     mode_saved = model.training
#     model.train(False)
#     use_cuda = torch.cuda.is_available()
#     # use_cuda = True
#     if use_cuda:
#         model.cuda()
#     outputs_full = []
#     labels_full = []

#     for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

#         img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
#             batch_sample['image_depth'], batch_sample[
#             'binary_label']

#         if use_cuda:
#             img_rgb = img_rgb.cuda()
#             img_ir = img_ir.cuda()
#             img_depth = img_depth.cuda()
#             target = target.cuda()

#         with torch.no_grad():

#             outputs_batch = model(img_rgb, img_depth, img_ir)
#             if isinstance(outputs_batch, tuple):
#                 outputs_batch = outputs_batch[0]
#             # print(outputs_batch)
#         outputs_full.append(outputs_batch)
#         labels_full.append(target)

#     model.train(mode_saved)
#     outputs_full = torch.cat(outputs_full, dim=0)
#     labels_full = torch.cat(labels_full, dim=0)
#     _, labels_predicted = torch.max(outputs_full.data, dim=1)
#     accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
#     # print((labels_full - labels_predicted))
#     accuracy = float("%.6f" % accuracy)

#     if hter:
#         predict_arr = np.array(labels_predicted.cpu())
#         label_arr = np.array(labels_full.cpu())

#         living_wrong = 0  # living -- spoofing
#         living_right = 0
#         spoofing_wrong = 0  # spoofing ---living
#         spoofing_right = 0

#         for i in range(len(predict_arr)):
#             if predict_arr[i] == label_arr[i]:
#                 if label_arr[i] == 1:
#                     living_right += 1
#                 else:
#                     spoofing_right += 1
#             else:
#                 # 错误
#                 if label_arr[i] == 1:
#                     living_wrong += 1
#                 else:
#                     spoofing_wrong += 1

#         try:

#             APCER = living_wrong / (living_wrong + living_right)
#             NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)
#             ACER = (APCER + NPCER) / 2

#             APCER = float("%.6f" % APCER)
#             NPCER = float("%.6f" % NPCER)
#             ACER = float("%.6f" % ACER)
#             accuracy = float("%.6f" % accuracy)
#         except Exception as e:
#             print(living_right, living_wrong, spoofing_right, spoofing_wrong), labels_predicted
#             return [accuracy, 0, 0, 0], labels_predicted

#         print(living_right, living_wrong, spoofing_right, spoofing_wrong)
#         return [accuracy, APCER, NPCER, ACER], labels_predicted
#     else:
#         return [accuracy], labels_predicted

def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output = model(img_rgb, img_depth, img_ir)
            if isinstance(output, tuple):
                output = output[0]

            loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()

            optimizer.step()

        # testing
        # result_test, _ = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        result_test, _ = calc_accuracy_kd_patch_feature(model=model, args=args, loader=test_loader, hter=True)

        accuracy_test = result_test[0]
        acer_test = result_test[-1]
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)

        if acer_test < acer_best and epoch > 30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path)

        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        # if epoch % save_interval == 0:
        #     train_state = {
        #         "Epoch": epoch,
        #         "model_state": model.state_dict(),
        #         "optim_state": optimizer.state_dict(),
        #         "args": args
        #     }
        #     models_dir = args.model_root + '/' + args.name + str(epoch) + '.pt'
        #     torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi_auxi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            fusion_loss = cost(output, target)

            x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

            # print(x_rgb_loss_batch.shape,p.shape)
            # print(p)

            x_rgb_loss = torch.sum(x_rgb_loss_batch * p[:, 0]) / p.shape[0]

            x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)
            x_ir_loss = torch.sum(x_ir_loss_batch * p[:, 1]) / p.shape[0]

            x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
            x_depth_loss = torch.sum(x_depth_loss_batch * p[:, 2]) / p.shape[0]

            loss = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            train_loss += loss.item()
            loss.backward()

            # if batch_num>10:
            #     print("weight.grad:", model.special_bone_rgb[0].weight.grad.mean(), model.special_bone_rgb[0].weight.grad.min(), model.special_bone_rgb[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_ir[0].weight.grad.mean(), model.special_bone_ir[0].weight.grad.min(), model.special_bone_ir[0].weight.grad.max())
            #     print("weight.grad:", model.special_bone_depth[0].weight.grad.mean(), model.special_bone_depth[0].weight.grad.min(), model.special_bone_depth[0].weight.grad.max())

            if (model.special_bone_rgb[0].weight.grad is None) or (model.special_bone_ir[0].weight.grad is None) or (
                    model.special_bone_depth[0].weight.grad is None):
                print("none!!!!!!none!!!!!")

            # print(model.special_bone_rgb[0].weight.grad)
            # print(model.special_bone_ir[0].weight.grad)
            # print(model.special_bone_depth[0].weight.grad)

            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)

        save_path = os.path.join(args.model_root, args.name + "_epoch_" + str(epoch)
                                 + '.pth')
        torch.save(model.state_dict(), save_path)

        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_auxi(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        optim.lr_scheduler.ExponentialLR
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_layer3, student_layer4, x_rgb_out, x_ir_out, x_depth_out, p = student_model(
                img_rgb, img_ir, img_depth)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            #     # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # patch_loss = patch_loss.cuda()
            #     # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            #     # whole_loss = whole_loss.cuda()
            #     # kd_loss = patch_loss + whole_loss
            #     kd_logits_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            #     kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)

            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            kd_feature_loss_1 = sp_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_2

            # 分类损失

            fusion_loss = criterionCls(student_whole_out, target)

            x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)

            x_rgb_loss = torch.sum(x_rgb_loss_batch * ((p[:, 0]) * (1 - p[:, 1]) * (1 - p[:, 2]))) / p.shape[0]

            x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)
            x_ir_loss = torch.sum(x_ir_loss_batch * ((p[:, 1]) * (1 - p[:, 0]) * (1 - p[:, 2]))) / p.shape[0]

            x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
            x_depth_loss = torch.sum(x_depth_loss_batch * ((p[:, 2]) * (1 - p[:, 0]) * (1 - p[:, 1]))) / p.shape[0]

            cls_loss = fusion_loss + x_rgb_loss + x_depth_loss + x_ir_loss

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    # train_duration_sec = int(time.time() - start)
    # print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_auxi_weak(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']



    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save
    auxi_loss_sum=0

    modality_combination = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    disttribution_distance = [0] * (len(modality_combination) - 1)
    weak_combintaions = []

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train

    while epoch < epoch_num:
        student_model.p = [0, 0, 0]
        start = datetime.datetime.now()
        fusion_loss_weak_list = []
        fuse_loss_strong_list = []
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_depth, img_ir)

            student_whole_out, student_layer3, student_layer4, x_rgb_out, x_ir_out, x_depth_out, p = student_model(
                img_rgb, img_depth, img_ir)



            kd_feature_loss = criterionKD(student_layer4, teacher_layer4,teacher_whole_out)


            fusion_loss = criterionCls(student_whole_out, target)


            #Eq14 in paper
            if epoch > args.begin_epoch:
                x_rgb_loss_batch = auxi_cross_entropy(x_rgb_out, target)
                weak_index = p == weak_combintaions[0].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).float()
                weak_index = weak_index.all(dim=1)

                x_rgb_loss = torch.sum(x_rgb_loss_batch*weak_index) / p.shape[0]

                x_ir_loss_batch = auxi_cross_entropy(x_ir_out, target)
                weak_index = p == weak_combintaions[1].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).float()
                weak_index = weak_index.all(dim=1)


                x_ir_loss = torch.sum(x_ir_loss_batch*weak_index) / p.shape[0]

                x_depth_loss_batch = auxi_cross_entropy(x_depth_out, target)
                weak_index = p == weak_combintaions[2].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).float()
                weak_index = weak_index.all(dim=1)

                x_depth_loss = torch.sum(x_depth_loss_batch*weak_index) / p.shape[0]


            if epoch > args.begin_epoch:
                cls_loss = fusion_loss + (x_rgb_loss + x_depth_loss + x_ir_loss) * args.lambda_kd_mar+ kd_feature_loss * args.lambda_kd_feature

            else:
                cls_loss = fusion_loss

            loss = cls_loss.cuda()


            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            if epoch>5:
                auxi_loss_sum+=(x_rgb_loss + x_depth_loss + x_ir_loss).item()
            loss.backward()
            optimizer.step()

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]

        # contrastive_ranking

        if epoch > 1 and epoch < args.begin_epoch:
            label_disttribution = []

            # EQ.9 in paper
            for j in range(len(modality_combination)):
                args.p = modality_combination[j]
                print(args.p)
                student_model.p = modality_combination[j]
                result_test, label_predict = calc_accuracy_kd_patch_feature(model=student_model, args=args,
                                                                            loader=test_loader,
                                                                            hter=True)


                # EQ10 & 11 in paper
                label_predict_hist = get_dataself_hist(np.array(label_predict.cpu()))
                print(result_test)
                print(label_predict_hist)
                v_list = [0] * args.class_num
                for k, v in label_predict_hist.items():
                    v_list[int(k)] = v
                v_arr = np.array(v_list)
                v_arr = v_arr / (np.sum(v_arr))
                label_disttribution.append([list(v_arr)])


            # EQ.12 13 in paper
            label_disttribution = torch.tensor(label_disttribution).float()
            if epoch > 1:
                for i in range(len(label_disttribution) - 1):
                    distance = F.kl_div(
                        F.log_softmax(label_disttribution[len(label_disttribution) - 1], dim=1),
                        F.softmax(label_disttribution[i], dim=1), reduction='batchmean')
                    print(distance)
                    disttribution_distance[i] = disttribution_distance[i] + distance


            print(np.array(disttribution_distance))


        # calculate the weak combinations
        if epoch == args.begin_epoch:
            max_index = disttribution_distance.index(max(disttribution_distance))
            weak_combintaion = modality_combination[max_index]
            print(weak_combintaion)
            weak_combintaions.append(weak_combintaion)
            print(weak_combintaions)
            one_indexs = [i for i in range(len(weak_combintaion)) if weak_combintaion[i] == 1]

            for index in one_indexs:
                zero_list = [0, 0, 0]
                zero_list[index] = 1
                weak_combintaions.append(zero_list)

            print(weak_combintaions)
            weak_combintaions = torch.tensor(weak_combintaions).cuda()

        if acer_test < acer_best and epoch > 30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 30:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), auxi_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0
        auxi_loss_sum=0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1


def train_knowledge_distill_patch_feature(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP, DAD
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        dad_loss = DAD().cuda()
    else:
        dad_loss = DAD()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6),
                                                                              np.int(args.train_epoch * 4 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        start = datetime.datetime.now()

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            student_whole_out, student_layer3, student_layer4 = student_model(img_rgb, img_ir, img_depth)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # logits蒸馏损失
            # if args.kd_mode in ['logits', 'st']:
            #     # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # patch_loss = patch_loss.cuda()
            #     # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
            #     # whole_loss = whole_loss.cuda()
            #     # kd_loss = patch_loss + whole_loss
            #     kd_logits_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
            #     # kd_logits_loss = bce_loss(student_patch_out, teacher_patch_out.detach())
            #     kd_logits_loss = kd_logits_loss.cuda()
            # else:
            #     kd_logits_loss = 0
            #     print("kd_Loss error")

            # feature 蒸馏损失
            # student_layer3 = torch.mean(student_layer3, dim=1)
            # teacher_layer3 = torch.mean(teacher_layer3, dim=1)
            # kd_feature_loss = mse_loss(student_layer3, teacher_layer3)

            # student_layer3=torch.unsqueeze(student_layer3,dim=1)
            kd_feature_loss_1 = sp_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = sp_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_2

            # print(kd_feature_loss.shape)

            # if args.margin:
            #
            #     teacher_whole_out_prob = F.softmax(teacher_whole_out, dim=1)
            #     H_teacher = torch.sum(-teacher_whole_out_prob * torch.log(teacher_whole_out_prob), dim=1)
            #     # print(H_teacher.shape)
            #     # H_teacher_prob = F.softmax(H_teacher * 64, dim=0)
            #     H_teacher_prob = H_teacher / torch.sum(H_teacher)
            #     kd_feature_loss = torch.sum(kd_feature_loss * H_teacher_prob)
            # else:
            #     kd_feature_loss = torch.mean(torch.sum(kd_feature_loss,dim=1))

            # print(H_teacher_prob)

            # 分类损失
            cls_loss = criterionCls(student_whole_out, target)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss * args.lambda_kd_feature

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            # kd_logits_loss_sum += kd_logits_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    # train_duration_sec = int(time.time() - start)
    # print("training is end", train_duration_sec)


def train_knowledge_distill_patch_feature_cefa(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP
    from loss.kd.at import AT
    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()

    if torch.cuda.is_available():
        pkt_loss = PKTCosSim().cuda()
    else:
        pkt_loss = PKTCosSim()

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    if torch.cuda.is_available():
        at_loss = AT(p=2).cuda()
    else:
        at_loss = AT(p=2)

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, multi_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                multi_sample['image_depth'], multi_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
            label = target

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir,
                                                                                                 img_depth)

            student_whole_out, student_patch_out, student_layer3, student_layer4 = student_model(img_rgb)

            kd_feature_loss_1 = at_loss(student_layer3, teacher_layer3)
            kd_feature_loss_2 = at_loss(student_layer4, teacher_layer4)
            kd_feature_loss = kd_feature_loss_1 + kd_feature_loss_2

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_feature_loss

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 15:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 15:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_logits_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        kd_logits_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)



def train_RAML(model, optimizer, train_loader, test_loader, args):

    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()
    if args.cuda:
        loss_ce = torch.nn.CrossEntropyLoss().cuda()
        loss_mae = torch.nn.L1Loss().cuda()
    else:
        loss_ce = torch.nn.CrossEntropyLoss()
        loss_mae = torch.nn.L1Loss()     

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 3 / 10),
                                                                              np.int32(args.train_epoch * 7 / 10),
                                                                              np.int32(args.train_epoch * 9 / 10)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    Loss_cls = 0
    Loss_uni = 0
    Loss_inf = 0
    epoch = 1
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
                label_mae = torch.zeros(target.size(0), 512).cuda()

            # optimizer.zero_grad()
            for p in model.parameters():
                p.grad = None

            model.args.epoch = epoch
            output, multi_ctr, rgb_out, ir_out, depth_out,  p = model(img_rgb, img_depth, img_ir)
            if isinstance(output, tuple):
                output = output[0]

            loss_cls = loss_ce(output, target)
        
            loss_cls_rgb = loss_ce(rgb_out, target)
            loss_cls_ir = loss_ce(ir_out, target)
            loss_cls_depth = loss_ce(depth_out, target)

            loss_uni = loss_cls_rgb + loss_cls_ir + loss_cls_depth
            
            loss_inf = loss_mae(multi_ctr[0], label_mae) + loss_mae(multi_ctr[1], label_mae) + loss_mae(multi_ctr[2], label_mae)
        
            loss = loss_cls + args.weight_uni * loss_uni + args.weight_inf * loss_inf


            Loss_cls += loss_cls.item()
            Loss_inf += loss_inf.item()
            Loss_uni += loss_uni.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # testing
        result_test, _ = calc_accuracy_kd_patch_feature(model=model, args=args, loader=test_loader, hter=True)


        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(Loss_cls / len(train_loader))
        log_list.append(Loss_inf / len(train_loader))
        log_list.append(Loss_uni / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss_cls={:.5f}, loss_uni={:.5f}, loss_inf={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        Loss_cls / len(train_loader),
                                                                                        Loss_uni / len(train_loader),
                                                                                        Loss_inf / len(train_loader),
                                                                                            accuracy_test, accuracy_best))
        Loss_cls = 0
        Loss_uni = 0
        Loss_inf = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step()
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])


        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)





def train_uncl_patch_feature(net_dict, cost_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']
    criterionRe = cost_dict['criterionRe']



    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 3 / 10),
                                                                              np.int32(args.train_epoch * 7 / 10),
                                                                              np.int32(args.train_epoch * 9 / 10)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train

    while epoch < epoch_num:
        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue


            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, _, teacher_layer = teacher_model(img_rgb, img_depth, img_ir)

            student_whole_out, _, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, p = student_model(
                                                                                                                               img_rgb, img_depth, img_ir)

            # if epoch < 30:
            #     teacher_layer = teacher_layer_3

            kd_feature_loss = criterionKD(rgb_mu_full, teacher_layer.detach(), rgb_logvar_full) + \
                              criterionKD(ir_mu_full, teacher_layer.detach(), ir_logvar_full) + \
                              criterionKD(depth_mu_full, teacher_layer.detach(), depth_logvar_full)   
            
            kd_shared_loss = criterionRe(rgb_mu_full, rgb_logvar_full, depth_mu_full, depth_logvar_full, ir_mu_full, ir_logvar_full, teacher_layer.detach())
                              


            fusion_loss = criterionCls(student_whole_out, target)


            if epoch > args.begin_epoch:
                cls_loss = fusion_loss + kd_feature_loss * args.lambda_kd_feature + kd_shared_loss * args.lambda_kd_shared
            else:
                cls_loss = fusion_loss + kd_feature_loss * args.lambda_kd_feature + kd_shared_loss * args.lambda_kd_shared

            loss = cls_loss.cuda()


            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.5)
            optimizer.step()

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best and epoch>args.save_interval:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch>args.save_interval:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(acer_test)
        log_list.append(acer_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f} acer_test={:.5f},  acer_best={:.5f}".format(epoch,
                                                                                                                            train_loss / len(
                                                                                                                            train_loader),
                                                                                                                            accuracy_test, accuracy_best,
                                                                                                                            acer_test, acer_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()
        
        print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1



def train_cl_patch_feature(net_dict, cost_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']
    criterionRe = cost_dict['criterionRe']



    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, T_max=args.train_epoch + 20, eta_min=args.lr*1e-2)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 3 / 10),
                                                                              np.int32(args.train_epoch * 7 / 10),
                                                                              np.int32(args.train_epoch * 9 / 10)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save
    auxi_loss_sum=0


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train

    while epoch < epoch_num:
        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']
            if epoch == 0:
                continue


            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            teacher_whole_out, _, teacher_layer = teacher_model(img_rgb, img_depth, img_ir)

            student_whole_out, rgb_mu_full, ir_mu_full, depth_mu_full, p = student_model(img_rgb, img_depth, img_ir)


            kd_feature_loss = criterionKD(rgb_mu_full, teacher_layer) + \
                              criterionKD(ir_mu_full, teacher_layer) + \
                              criterionKD(depth_mu_full, teacher_layer)   

            kd_shared_loss = criterionRe(rgb_mu_full, depth_mu_full, ir_mu_full, teacher_layer.detach())
                              
            fusion_loss = criterionCls(student_whole_out, target)


            if epoch > args.begin_epoch:
                cls_loss = fusion_loss + kd_feature_loss * args.lambda_kd_feature + kd_shared_loss * args.lambda_kd_shared
            else:
                cls_loss = fusion_loss

            loss = cls_loss.cuda()


            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_feature_loss_sum += kd_feature_loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.5)
            optimizer.step()

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), auxi_loss_sum / (len(train_loader)),
              kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {},  loss={:.5f},  accuracy_test={:.5f},  accuracy_best={:.5f},  acer_test={:.5f},  acer_best={:.5f}".format(epoch,
                                                                                                                    train_loss / len(
                                                                                                                    train_loader),
                                                                                                                    accuracy_test, accuracy_best,
                                                                                                                    acer_test, acer_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        auxi_loss_sum=0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()
        
        print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1




def train_ETMC(net_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    # models_dir = '/remote-home/mengxichen/MMANet-CVPR2023-main/output/models/cefa_test-uncl-sigma-selfkd-64-dev-wofusion-avgFalse-topo-only-fbn-512-mean(0.1)-110-cat-woweight-2_version_3_kd_1.0_shared_1.0.pt'
    
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])

    # Training initialization
    epoch_num = args.train_epoch
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            
            epoch = state_read['Epoch'] + 1
            
              
    # Train

    while epoch < epoch_num:

        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

                
            student_whole_out, alpha, alpha_f, alpha_a = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch


            fusion_loss = student_model.ce_loss(target, alpha_a, 2, epoch, 10).mean()
            loss_CE_A = student_model.ce_loss(target, alpha[0], 2, epoch, 10).mean()
            loss_CE_L = student_model.ce_loss(target, alpha[1], 2, epoch, 10).mean()
            loss_CE_V = student_model.ce_loss(target, alpha[2], 2, epoch, 10).mean()
            loss_CE_F = student_model.ce_loss(target, alpha_f, 2, epoch, 10).mean()
            
            loss_CE = loss_CE_A + loss_CE_L + loss_CE_V + loss_CE_F
        
            loss = fusion_loss + args.weight_etmc * loss_CE


            train_loss += loss.item()
            cls_loss_sum += fusion_loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 10)
            optimizer.step()

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best and epoch>30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_best < accuracy_test and epoch>30:
            accuracy_best = accuracy_test


        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(acer_test)
        log_list.append(acer_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f} acer_test={:.5f},  acer_best={:.5f}".format(epoch,
                                                                                                                            train_loss / len(
                                                                                                                            train_loader),
                                                                                                                            accuracy_test, accuracy_best,
                                                                                                                            acer_test, acer_best))
        train_loss = 0
        cls_loss_sum = 0
        cos_scheduler.step()
        print(epoch, optimizer.param_groups[0]['lr'])

        
        train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
        models_dir = args.model_root + '/' + args.name + '.pt'
        # save model and para


        if (epoch) % save_interval == 0 and epoch != args.total_epoch:
        # if (epoch-args.total_epoch) in [save_interval, save_interval*3]:
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1



def train_MD(net_dict, cost_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    # models_dir = '/remote-home/mengxichen/MMANet-CVPR2023-main/output/models/cefa_test-uncl-sigma-selfkd-64-dev-wofusion-avgFalse-topo-only-fbn-512-mean(0.1)-110-cat-woweight-2_version_3_kd_1.0_shared_1.0.pt'
    
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']

    cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
    criterionCls = cost_dict['criterionCls']
    # Training initialization
    epoch_num = args.train_epoch
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            
            epoch = state_read['Epoch'] + 1
            
              
    # Train

    while epoch < epoch_num:

        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            student_whole_out, rgb_out, depth_out, ir_out, rgb_var_full, depth_var_full, ir_var_full, rgb_logvar, depth_logvar, ir_logvar = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch


            fusion_loss = criterionCls(student_whole_out, target)
            loss_CE_A =  criterionCls(rgb_out, target)
            loss_CE_L = criterionCls(ir_out, target)
            loss_CE_V = criterionCls(depth_out, target)
            
            loss_CE = loss_CE_A + loss_CE_L + loss_CE_V

            pred_A = F.softmax(rgb_out, dim=-1)
            A_target = torch.gather(input=pred_A, dim=1, index=target.unsqueeze(dim=1)).view(-1)
            pred_L = F.softmax(ir_out, dim=-1)
            L_target = torch.gather(input=pred_L, dim=1, index=target.unsqueeze(dim=1)).view(-1)
            pred_V = F.softmax(depth_out, dim=-1)
            V_target = torch.gather(input=pred_V, dim=1, index=target.unsqueeze(dim=1)).view(-1)      

            confidence_loss = torch.mean(F.mse_loss(rgb_var_full.view(-1), A_target))+torch.mean(F.mse_loss(ir_var_full.view(-1), L_target))+torch.mean(F.mse_loss(depth_var_full.view(-1), V_target))

            loss_INF = torch.mean(rgb_logvar) + torch.mean(depth_logvar) + torch.mean(ir_logvar)

            loss = fusion_loss + args.weight_sum * loss_CE + args.weight_kld * loss_INF + args.weight_con*confidence_loss


            train_loss += loss.item()
            cls_loss_sum += fusion_loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 10)
            optimizer.step()

        student_model.p = [0, 0, 0]
        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best and epoch>30:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_best < accuracy_test and epoch>30:
            accuracy_best = accuracy_test


        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(acer_test)
        log_list.append(acer_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f} acer_test={:.5f},  acer_best={:.5f}".format(epoch,
                                                                                                                            train_loss / len(
                                                                                                                            train_loader),
                                                                                                                            accuracy_test, accuracy_best,
                                                                                                                            acer_test, acer_best))
        train_loss = 0
        cls_loss_sum = 0
        cos_scheduler.step()
        print(epoch, optimizer.param_groups[0]['lr'])

        
        train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
        models_dir = args.model_root + '/' + args.name + '.pt'
        # save model and para


        if (epoch) % save_interval == 0 and epoch != args.total_epoch:
        # if (epoch-args.total_epoch) in [save_interval, save_interval*3]:
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1




def train_uncl_patch_feature_teacher(net_dict, cost_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    # if args.dataset=='surf':
    #     models_dir = '/DB/public/mengxichen/code/mengxichen/MMANet-CVPR2023-main/segmentation/results/cityscape/uncl-mad/cityscapes-with-depth/checkpoints_15_05_2024-21_32_53-836829/ckpt_epoch_150.pth'
    # else:
    #     models_dir = '/remote-home/mengxichen/MMANet-CVPR2023-main/output/models/cefa_uncl_version_1.pt'
    
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']
    criterionRe = cost_dict['criterionRe']


    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_logits_loss_sum = 0
    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    kd_loss_best = 1000
    log_list = []  # log need to save


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = args.total_epoch + 1
            optimizer.param_groups[0]['lr'] = args.lr
            print("retaining")
            teacher_model.load_state_dict(state_read['model_state'])   
            teacher_model.p = [1,1,1]
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False 

            
              
    # Train

    while epoch < epoch_num+1:

        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()


            # student_whole_out, fusion_mu, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, fusion_logvar, p = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch
            student_whole_out, fusion_mu, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, fusion_logvar, p = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch

            multi_index = p.squeeze().sum(-1) > 1


            fusion_loss = criterionCls(student_whole_out, target)


            if epoch > args.total_epoch:

                # teacher_whole_out, teacher_layer, _ ,_, _, _, _, _, _, _, _ = teacher_model(img_rgb, img_depth, img_ir)
                # teacher_whole_out, teacher_layer, _ ,_, _, _, _, _, _, _ = teacher_model(img_rgb, img_depth, img_ir)
                teacher_whole_out, teacher_layer, _, _, _, _, _, _, _, _ = teacher_model(img_rgb, img_depth, img_ir)

                # index = p.squeeze().sum(-1) < 3 
                kd_feature_loss = criterionKD(rgb_mu_full, teacher_layer.detach(), rgb_logvar_full, target, False) + \
                                  criterionKD(ir_mu_full, teacher_layer.detach(), ir_logvar_full, target, False) + \
                                  criterionKD(depth_mu_full, teacher_layer.detach(), depth_logvar_full, target, False) +\
                                  criterionKD(fusion_mu[multi_index], teacher_layer.detach(), fusion_logvar[multi_index], target, True, multi_index) 
                                #   criterionKD(fusion_mu[multi_index], teacher_layer.detach(), fusion_logvar[multi_index], target, False, multi_index) 
                            
                kd_shared_loss = criterionRe(rgb_mu_full, depth_mu_full, ir_mu_full, fusion_mu, multi_index, teacher_layer.detach(),target)
                # kld_loss = - 0.5 * torch.mean(torch.sum(1 + rgb_logvar_full - rgb_mu_full.pow(2) - rgb_logvar_full.exp(), dim=1)) - \
                #              0.5 * torch.mean(torch.sum(1 + ir_logvar_full - ir_mu_full.pow(2) - ir_logvar_full.exp(), dim=1)) - \
                #              0.5 * torch.mean(torch.sum(1 + depth_logvar_full - depth_mu_full.pow(2) - depth_logvar_full.exp(), dim=1)) - \
                #              0.5 * torch.mean(torch.sum(1 + fusion_logvar - fusion_mu.pow(2) - fusion_logvar.exp(), dim=1))    
                              
                cls_loss = fusion_loss + kd_feature_loss * args.lambda_kd_feature + kd_shared_loss * args.lambda_kd_shared
                
                kd_feature_loss_sum += kd_feature_loss.item()
                
            else:
                
                cls_loss = fusion_loss 

            loss = cls_loss.cuda()


            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 50.0)
            optimizer.step()

        student_model.p = [0, 0, 0]

        result_test, _, kd_loss_test = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best and epoch > args.total_epoch:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path)

        train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }

        models_dir = args.model_root + '/' + args.name + '.pt'
        # save model and para
        if kd_loss_test < kd_loss_best and epoch > 30 and epoch < args.total_epoch + 1:
        # if acer_test < acer_best:
            kd_loss_best = kd_loss_test
            torch.save(train_state, models_dir)


        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(acer_test)
        log_list.append(acer_best)
        log_list.append(kd_loss_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f} acer_test={:.5f},  acer_best={:.5f}, kd_best={:.5f}".format(epoch,
                                                                                                                            train_loss / len(
                                                                                                                            train_loader),
                                                                                                                            accuracy_test, accuracy_best,
                                                                                                                            acer_test, acer_best, kd_loss_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        print(epoch, optimizer.param_groups[0]['lr'])

        # mod = epoch % (save_interval + args.total_epoch)
        # if mod < args.total_epoch + 1:
        # if epoch < args.total_epoch + 1:
        #     lr = args.lr
        if epoch < args.total_epoch // 6:
            lr = args.lr
        elif epoch > (args.total_epoch // 6 - 1) and epoch < args.total_epoch * 2 // 6:
            lr = args.lr * 0.1
        elif epoch > (args.total_epoch * 2 // 6 - 1) and epoch < args.total_epoch * 3 // 6:
            lr = args.lr * 0.1**2
        elif epoch > (args.total_epoch * 3 // 6 - 1) and epoch < args.total_epoch + 1:
            lr = args.lr * 0.1**3
        else: 
            mod = (epoch - args.total_epoch) % save_interval
            # mod = mod - args.total_epoch
        # mod = epoch % save_interval
            lr = args.lr * 0.9 ** mod


        optimizer.param_groups[0]['lr'] = lr
        


        if epoch == args.total_epoch:
            state_read = torch.load(models_dir)
            teacher_model.load_state_dict(state_read['model_state'])
            teacher_model.eval()
            teacher_model.p = [1,1,1]
            for param in teacher_model.parameters():
                param.requires_grad = False
            if torch.cuda.is_available():
                teacher_model.cuda()  
            print("warmup loading")   

            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            optimizer.param_groups[0]['lr'] = args.lr


        # if (epoch-args.total_epoch) % save_interval == 0 and epoch != args.total_epoch:
        # # if epoch % (save_interval + args.total_epoch) == 0 and epoch != args.total_epoch:
        #     torch.save(train_state, models_dir)

        #     teacher_model.load_state_dict(train_state['model_state'])
        #     teacher_model.eval()
        #     teacher_model.p = [1,1,1]
        #     for param in teacher_model.parameters():
        #         param.requires_grad = False
        #     if torch.cuda.is_available():
        #         teacher_model.cuda()  
            # print("teacher loading")

        # if epoch == epoch_num:  
        #     path =  args.model_root + '/' + args.name + 'final.pt'
        #     torch.save(train_state, models_dir)


        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1




def train_uncl_patch_feature_teache_pre(net_dict, cost_dict, optimizer, train_loader, test_loader, args):

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']
    criterionRe = cost_dict['criterionRe']


    # Training initialization
    epoch_num = args.train_epoch

    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0

    kd_feature_loss_sum = 0
    epoch = 1
    accuracy_best = 0
    acer_best = 1
    log_list = []  # log need to save

              
    # Train

    while epoch < epoch_num+1:

        student_model.p = [0, 0, 0]

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_depth, img_ir, target = batch_sample['image_x'], batch_sample['image_depth'], \
                batch_sample['image_ir'], batch_sample[
                'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            # student_whole_out, _, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, fusion_mu, fusion_logvar, p = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch
            student_whole_out, fusion_mu, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full, fusion_logvar, p = student_model(img_rgb, img_depth, img_ir)#  epoch > args.total_epoch

            multi_index = p.squeeze().sum(-1) > 1

            fusion_loss = criterionCls(student_whole_out, target)



            # teacher_whole_out, teacher_layer, _ ,_, _, _, _, _, _, _, _ = teacher_model(img_rgb, img_depth, img_ir)
            teacher_whole_out, teacher_layer, _ ,_, _, _, _, _, _, _ = teacher_model(img_rgb, img_depth, img_ir)


            kd_feature_loss = criterionKD(rgb_mu_full, teacher_layer.detach(), rgb_logvar_full, target, False) + \
                                criterionKD(ir_mu_full, teacher_layer.detach(), ir_logvar_full, target, False) + \
                                criterionKD(depth_mu_full, teacher_layer.detach(), depth_logvar_full, target, False) +\
                                criterionKD(fusion_mu[multi_index], teacher_layer.detach(), fusion_logvar[multi_index], target, True, multi_index) 
                                
                
              
            kd_shared_loss = criterionRe(rgb_mu_full, depth_mu_full, ir_mu_full, fusion_mu, multi_index, teacher_layer.detach(),target)
 
                              
            cls_loss = fusion_loss + kd_feature_loss * args.lambda_kd_feature + kd_shared_loss * args.lambda_kd_shared
                
            kd_feature_loss_sum += kd_feature_loss.item()


            loss = cls_loss.cuda()


            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()

            loss.backward()

            optimizer.step()

        student_model.p = [0, 0, 0]

        result_test, _ = calc_accuracy_kd_patch_feature(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]

        acer_test = result_test[3]


        if acer_test < acer_best and epoch> epoch_num - save_interval*0.6:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )


        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(acer_test)
        log_list.append(acer_best)
        print(cls_loss_sum / len(train_loader), kd_feature_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f} acer_test={:.5f},  acer_best={:.5f}".format(epoch,
                                                                                                                            train_loss / len(
                                                                                                                            train_loader),
                                                                                                                            accuracy_test, accuracy_best,
                                                                                                                            acer_test, acer_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_feature_loss_sum = 0
        print(epoch, optimizer.param_groups[0]['lr'])


        mod = epoch % save_interval
        lr = args.lr * 0.9 ** mod
 
        optimizer.param_groups[0]['lr'] = lr
        

        # save model and para


        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1