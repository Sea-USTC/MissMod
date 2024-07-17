import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from models.resnet18_se_trm import resnet18_se_trm, resnet18_uni_trm
from lib.model_arch_utils import Flatten
import numpy as np
import random
from lib.model_arch import modality_drop, unbalance_modality_drop
import torch.nn.functional as F
# from models.lora import LoRAConv2d3modal, LoRA3Linear
from models.trm_decoder import TransformerDecoder3lora, TransformerDecoderLayer3lora


class SURF_Multi(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        layer4 = self.shared_bone[3](x)
        x = self.shared_bone[4](layer4)
        return x, layer3, layer4


class SURF_Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        # print("img_shape:")
        # print(img_rgb.shape, img_depth.shape, img_ir.shape)
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)
        # print("x_shape:")
        # print(x_rgb.shape, x_depth.shape, x_ir.shape)

        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        # print("drop_shape:")
        # print(x_rgb.shape, x_depth.shape, x_ir.shape)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4


class SURF_Baseline_Auxi(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 1024
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_ir_out = self.auxi_bone(x_ir)
        x_depth_out = self.auxi_bone(x_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)



        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p


class SURF_Baseline_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 1024
        self.transformer = nn.Conv2d(1024, 1024, 1, 1)
        self.transformer_rgb = nn.Conv2d(1024, 1024, 1, 1)
        self.transformer_depth = nn.Conv2d(1024, 1024, 1, 1)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_depth_out = self.auxi_bone(x_depth)

        x_rgb_trans = self.transformer(x_rgb)
        x_depth_trans = self.transformer(x_depth)

        x_rgb_depth = (x_rgb_trans + x_depth_trans) / 2
        x_rgb_depth = self.auxi_bone(x_rgb_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)


        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p


class SURF_Baseline_Auxi_Weak_Layer4(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )



    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)


        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

class SURF_MMANet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )



    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)



        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)

        # x_rgb_out = self.auxi_bone(x)
        # x_rgb_depth = self.auxi_bone(x)
        # x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, p


class SURF_MV(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]



        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4


class SURF_MV_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)

        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        if args.buffer:
            self.auxi_bone = nn.Sequential(
                nn.Conv2d(args.inplace_new, args.inplace_new, 1, 1),
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )
        else:
            self.auxi_bone = nn.Sequential(
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]


        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p




class SURF_RAMLNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))

        self.shared_bone = nn.Sequential(model_resnet18_se_1.fc)
        
        self.net_rgb = nn.Sequential(nn.Linear(512, 1024),
                                     nn.Dropout(0.5))
        self.net_ir = nn.Sequential(nn.Linear(512, 1024),
                                     nn.Dropout(0.5))
        self.net_depth = nn.Sequential(nn.Linear(512, 1024),
                                     nn.Dropout(0.5))
    def reparametrize(self, mu, logvar, k=1):
        eps = torch.randn(mu.size(0), k, mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(0.5*torch.exp(logvar.unsqueeze(1))).add_(mu.unsqueeze(1)).squeeze()
        return samples

    def infer(self, rgb, ir, depth, missing_index): 
        rgb_dis = self.net_rgb(rgb)
        mu     = rgb_dis[:,:512].unsqueeze(1)
        logvar = rgb_dis[:,512:].unsqueeze(1)

        depth_dis = self.net_depth(depth)
        mu     = torch.cat((mu, depth_dis[:,:512].unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, depth_dis[:,512:].unsqueeze(1)), dim=1)

        ir_dis = self.net_ir(ir)
        mu     = torch.cat((mu, ir_dis[:,:512].unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, ir_dis[:,512:].unsqueeze(1)), dim=1)

        mu, weight = self.ProductOfExperts(mu, logvar, missing_index)
        
        return mu, weight, rgb_dis[:,:512], rgb_dis[:,512:], ir_dis[:,:512], ir_dis[:,512:], depth_dis[:,:512], depth_dis[:,512:]
    

    def ProductOfExperts(self, mu, logvar, missing_index, eps=1e-8):
        var       = torch.exp(logvar) + eps
        T         = 1. / var
        weight    = T  / torch.sum(T, dim=1).unsqueeze(1)
        missing_index = missing_index.squeeze()
        pd_mu     = torch.sum(mu * T * missing_index.unsqueeze(2), dim=1) / torch.sum(T * missing_index.unsqueeze(2), dim=1)

        return pd_mu, weight

    def modality_drop(self, x_rgb, x_depth,x_ir, p, args):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()

        x_rgb = x_rgb * p[:, 0]
        x_depth = x_depth * p[:, 1]
        x_ir = x_ir * p[:, 2]

        return x_rgb, x_depth,x_ir, p


    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)


        x_rgb_miss, x_depth_miss, x_ir_miss, p = self.modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)


        miss_feat, _, _, _, _, _, _, _ = self.infer(x_rgb_miss, x_ir_miss, x_depth_miss, p)
    

        full_feat, weight, rgb_mu_full, rgb_logvar_full, ir_mu_full, ir_logvar_full, depth_mu_full, depth_logvar_full = self.infer(x_rgb, x_ir, x_depth, torch.ones(p.size()).cuda()) 

        rgb_feat = self.reparametrize(rgb_mu_full, rgb_logvar_full)
        ir_feat = self.reparametrize(ir_mu_full, ir_logvar_full) 
        depth_feat = self.reparametrize(depth_mu_full, depth_logvar_full) 
           
        rgb_out = self.shared_bone(rgb_feat)
        ir_out = self.shared_bone(ir_feat)
        depth_out = self.shared_bone(depth_feat)

        miss_out = self.shared_bone(miss_feat)

        # if torch.isnan((miss_out + rgb_out + ir_out + depth_out).sum()):
        #     k = 1

        rgb_ctr = rgb_mu_full * weight[:,0,:] 
        ir_ctr = ir_mu_full * weight[:,2,:] 
        depth_ctr = depth_mu_full * weight[:,1,:]

        multi_ctr = (rgb_ctr, ir_ctr, depth_ctr) 

        return miss_out, multi_ctr, rgb_out, ir_out, depth_out,  p



class SURF_UNCLNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer4_new,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc
                                         )

        self.net_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     Flatten(1),
                                     nn.Linear(512, 512),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Linear(512,512))
        
        self.net_ir = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     Flatten(1),
                                     nn.Linear(512, 512),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Linear(512,512))
        
        self.net_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       Flatten(1),
                                       nn.Linear(512, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Linear(1024,512))

        self.net_shared = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       Flatten(1),
                                       nn.Linear(512, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Linear(512,512))

        # self.mus = nn.Parameter(torch.zeros([1,3,1,1,1]), requires_grad=False)
        # self.betas = nn.Parameter(torch.ones([1,3,1,1,1]), requires_grad=False)

    def reparametrize(self, mu, logvar, k=1):
        eps = torch.randn((mu.size(0), k, mu.size(1), mu.size(2), mu.size(3)), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1).unsqueeze(3).unsqueeze(4))).add_(mu.unsqueeze(1)).squeeze()
        return samples

    def infer(self, rgb, ir, depth, missing_index): 
        rgb_logvar = self.net_rgb(rgb)
        mu     = rgb.unsqueeze(1)
        logvar = rgb_logvar.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        depth_logvar = self.net_depth(depth)
        mu     = torch.cat((mu, depth.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, depth_logvar.unsqueeze(1).unsqueeze(3).unsqueeze(4)), dim=1)

        ir_logvar = self.net_ir(ir)
        mu     = torch.cat((mu, ir.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, ir_logvar.unsqueeze(1).unsqueeze(3).unsqueeze(4)), dim=1)


        mu = self.Adaptive_Weight(mu, logvar, missing_index)
        
        return mu, rgb_logvar, depth_logvar, ir_logvar
    

    def Adaptive_Weight(self, mu, logvar, missing_index, eps=1e-12):
        logvar_clone = logvar.clone().detach().requires_grad_(True)
        
        var = torch.exp(logvar_clone) + eps
        T = 1. / var
        # T.mean(2).mean(0).unsqueeze(0).unsqueeze(2) * 0.5 + self.mus * 0.5 
        # weight_self = torch.sigmoid((T - self.mus)/(self.betas.pow(2) + eps).sqrt())
        # mu = mu * weight_self
        pd_mu = (mu * T * missing_index) / torch.sum(T * missing_index, dim=1).unsqueeze(1)

        return pd_mu


    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)


        x_rgb_miss, x_depth_miss, x_ir_miss, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)


        miss_feat, _, _, _, = self.infer(x_rgb_miss, x_ir_miss, x_depth_miss, p)
    

        full_feat, rgb_logvar_full, depth_logvar_full, ir_logvar_full = self.infer(x_rgb, x_ir, x_depth, torch.ones(p.size()).cuda()) 

        # rgb_feat = self.reparametrize(x_rgb, rgb_logvar_full)
        # ir_feat = self.reparametrize(x_ir, ir_logvar_full) 
        # depth_feat = self.reparametrize(x_depth, depth_logvar_full) 

        x = torch.cat((miss_feat[:,0,:,:,:], miss_feat[:,1,:,:,:], miss_feat[:,2,:,:,:]), dim=1)

        layer_4 = self.shared_bone[0](x)
        shared_logvar = self.net_shared(layer_4)
        # layer_4_new = layer_4 * torch.sigmoid((1./(torch.exp(shared_logvar.unsqueeze(2).unsqueeze(3)) + 1e-12) - self.mus)/(self.betas.pow(2) + 1e-12).sqrt())
        miss_out = self.shared_bone[1](layer_4)
        miss_out = self.shared_bone[2](miss_out)
        miss_out = self.shared_bone[3](miss_out)

        # if torch.isnan((miss_out + rgb_out + ir_out + depth_out).sum()):
        #     k = 1

        return miss_out, layer_4, shared_logvar, x_rgb, rgb_logvar_full, x_ir, ir_logvar_full, x_depth, depth_logvar_full, p
    







class SURF_UNCLBaseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.avg = True

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))
        
        self.shared_bone = nn.Sequential(nn.Linear(512*3, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         model_resnet18_se_1.fc)


    def modality_drop(self, x_rgb, x_depth,x_ir, p, args):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()

        x_rgb = x_rgb * p[:, 0]
        x_depth = x_depth * p[:, 1]
        x_ir = x_ir * p[:, 2]

        return x_rgb, x_depth,x_ir, p

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)
        x_rgb, x_depth, x_ir, p = self.modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)

        layer4 = self.shared_bone[0](x)
        x = self.shared_bone[1](layer4)
        x = self.shared_bone[2](x)
        x = self.shared_bone[3](x)

        # print(x.shape)
        return x, layer4




class SURF_CLNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.avg = False

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))

                
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 =  model_resnet18_se_1.fc

    def modality_drop(self, p, batchsize):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * batchsize]

            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()

        return p



    def unbalance_modality_drop(self, p, batchsize, prob):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]
        mode_num = 7

        if p == [0, 0, 0]:
            p = []
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            p = [p * batchsize]
            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
    
        p = p.float().cuda()

        return p

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)


        missing_index =  self.modality_drop(self.p, x_rgb.shape[0])

        x_rgb_miss = x_rgb * missing_index[:, 0]
        x_depth_miss = x_depth * missing_index[:, 1]
        x_ir_miss = x_ir * missing_index[:, 2]

        x = torch.cat((x_rgb_miss/3, x_depth_miss/3, x_ir_miss/3), dim=1)

        layer_student = self.fc1(x)

        miss_out = self.fc2(layer_student)


        return miss_out, layer_student, x_rgb, x_ir, x_depth, missing_index





# class SURF_UNCLLateNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         args.inplace_new = 384
#         model_resnet18_se_1 = resnet18_se(args, pretrained=False)
#         model_resnet18_se_2 = resnet18_se(args, pretrained=False)
#         model_resnet18_se_3 = resnet18_se(args, pretrained=False)
#         self.p = args.p
#         self.drop_mode = args.drop_mode
#         self.args = args
#         self.avg = False

#         self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
#                                               model_resnet18_se_1.bn1,
#                                               model_resnet18_se_1.relu,
#                                               model_resnet18_se_1.maxpool,
#                                               model_resnet18_se_1.layer1,
#                                               model_resnet18_se_1.layer2,
#                                               model_resnet18_se_1.se_layer,
#                                               model_resnet18_se_1.layer3,
#                                               model_resnet18_se_1.layer4,
#                                               model_resnet18_se_1.avgpool,
#                                               Flatten(1))
#         self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
#                                              model_resnet18_se_2.bn1,
#                                              model_resnet18_se_2.relu,
#                                              model_resnet18_se_2.maxpool,
#                                              model_resnet18_se_2.layer1,
#                                              model_resnet18_se_2.layer2,
#                                              model_resnet18_se_2.se_layer,
#                                              model_resnet18_se_2.layer3,
#                                              model_resnet18_se_2.layer4,
#                                              model_resnet18_se_2.avgpool,
#                                              Flatten(1))
#         self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
#                                                 model_resnet18_se_3.bn1,
#                                                 model_resnet18_se_3.relu,
#                                                 model_resnet18_se_3.maxpool,
#                                                 model_resnet18_se_3.layer1,
#                                                 model_resnet18_se_3.layer2,
#                                                 model_resnet18_se_3.se_layer,
#                                                 model_resnet18_se_3.layer3,
#                                                 model_resnet18_se_3.layer4,
#                                                 model_resnet18_se_3.avgpool,
#                                                 Flatten(1))

        
#         self.net_rgb = nn.Sequential(nn.Linear(512, 512),
#                                     #  nn.ReLU(inplace=True),
#                                     nn.LeakyReLU(0.1),
#                                     #  nn.Dropout(0.5),
#                                      nn.Linear(512, 512))
        
#         self.net_ir =  nn.Sequential(nn.Linear(512, 512),
#                                     #  nn.ReLU(inplace=True),
#                                     nn.LeakyReLU(0.1),
#                                     #  nn.Dropout(0.5),
#                                      nn.Linear(512, 512))
        
#         self.net_depth = nn.Sequential(nn.Linear(512, 512),
#                                     #    nn.ReLU(inplace=True),
#                                        nn.LeakyReLU(0.1),
#                                     #    nn.Dropout(0.5),
#                                        nn.Linear(512, 512))
        
        
#         self.shared_bone = nn.Sequential(nn.Linear(512, 512),
#                                         #  nn.ReLU(inplace=True),
#                                         nn.LeakyReLU(0.1),
#                                          nn.Dropout(0.5),
#                                         #  nn.Linear(512, 512),
#                                          model_resnet18_se_1.fc)
        
#     def reparametrize(self, mu, logvar, k=1):
#         eps = torch.randn((mu.size(0), k, mu.size(1)), dtype=mu.dtype, device=mu.device)
#         samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(mu.unsqueeze(1)).squeeze()
#         return samples
    

#     def Adaptive_Weight(self, mu, logvar, missing_index, eps=1e-12): 
#         # logvar_clone = logvar.clone().detach().requires_grad_(True)
#         # var = torch.exp(logvar_clone) + eps
#         var = torch.exp(logvar) + eps

#         T = 1. / var 
#         missing_index = missing_index.squeeze()
#         pd_mu = (mu * T * missing_index.unsqueeze(2)) / torch.sum(T * missing_index.unsqueeze(2), dim=1).unsqueeze(1)
#         pd_sigma = torch.log(1 / torch.sum(T * missing_index.unsqueeze(2), dim=1)+eps)
        
#         return pd_mu, pd_sigma
    

#     def infer(self, rgb, ir, depth, p): 
#         rgb_logvar = self.net_rgb(rgb)
#         mu     = rgb.unsqueeze(1)
#         logvar = rgb_logvar.unsqueeze(1)

#         depth_logvar = self.net_depth(depth)
#         mu     = torch.cat((mu, depth.unsqueeze(1)), dim=1)
#         logvar = torch.cat((logvar, depth_logvar.unsqueeze(1)), dim=1)

#         ir_logvar = self.net_ir(ir)
#         mu     = torch.cat((mu, ir.unsqueeze(1)), dim=1)
#         logvar = torch.cat((logvar, ir_logvar.unsqueeze(1)), dim=1)

#         missing_index =  self.modality_drop(p, rgb_logvar.shape[0])

#         mu, pd_sigma = self.Adaptive_Weight(mu, logvar, missing_index)
#         mu_full, _ = self.Adaptive_Weight(mu, logvar, torch.ones(missing_index.size()).cuda())
        
#         return mu, rgb_logvar, depth_logvar, ir_logvar, pd_sigma, missing_index


#     def modality_drop(self, p, batchsize):
#         modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
#         index_list = [x for x in range(7)]

#         if p == [0, 0, 0]:
#             p = []

#             prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
#             for i in range(batchsize):
#                 index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
#                 p.append(modality_combination[index])

#             p = np.array(p)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)

#         else:
#             p = p
#             # print(p)
#             p = [p * batchsize]

#             p = np.array(p).reshape(batchsize, 3)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)


#             # print(p[:, 0], p[:, 1], p[:, 2])
#         p = p.float().cuda()


#         return p


#     def forward(self, img_rgb, img_depth, img_ir, is_train=True):

#         x_rgb = self.special_bone_rgb(img_rgb)
#         x_ir = self.special_bone_ir(img_ir)
#         x_depth = self.special_bone_depth(img_depth) 

#         miss_feat, rgb_logvar_full, depth_logvar_full, ir_logvar_full, pd_sigma, missing_index = self.infer(x_rgb, x_ir, x_depth, self.p)
        

#         if self.avg == True:
#             p = self.modality_drop(self.p, x_depth.shape[0])
#             x_rgb_miss = x_rgb * p[:, 0]
#             x_depth_miss = x_depth * p[:, 1]
#             x_ir_miss = x_ir * p[:, 2]

#             x = x_rgb_miss/3 + x_depth_miss/3 + x_ir_miss/3
#             pd_sigma = None

#         else:     
            
#             x = miss_feat[:,0,:]+ miss_feat[:,1,:]+miss_feat[:,2,:]


#         layer_student = self.shared_bone[0](x)
#         # layer_full = self.shared_bone[0](x_full)

#         miss_out = self.shared_bone[1](layer_student)
#         miss_out = self.shared_bone[2](miss_out)
#         miss_out = self.shared_bone[3](miss_out)

#         # if torch.isnan((miss_out + rgb_out + ir_out + depth_out).sum()):
#         #     k = 1

#         return miss_out, layer_student, x_rgb, rgb_logvar_full, x_ir, ir_logvar_full, x_depth, depth_logvar_full, x, pd_sigma, missing_index
        




class SURF_ETMCNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))

        
        self.net_rgb = model_resnet18_se_1.fc
        
        self.net_ir = model_resnet18_se_2.fc
        
        self.net_depth = model_resnet18_se_3.fc
        
        
        self.shared_bone = nn.Sequential(nn.Linear(512*3, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 2))
        
    

    def modality_drop(self, p, batchsize):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * batchsize]

            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()


        return p.squeeze()

    def infer(self, rgb, ir, depth): 
        pred = dict() 
        
        pred[0] = F.softplus(self.net_rgb(rgb))
        pred[1] = F.softplus(self.net_depth(depth))
        pred[2] = F.softplus(self.net_ir(ir))
        x = torch.cat((rgb, depth, ir),dim=-1)
        full_pred = F.softplus(self.shared_bone(x))

        
        
        return pred, full_pred


    def KL(self, alpha, c):
        beta = torch.ones((1, c)).cuda()
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl


    def ce_loss(self, p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

        annealing_coef = min(1, global_step / annealing_step)

        alp = E * (1 - label) + 1
        B = annealing_coef * self.KL(alp, c)

        return (A + B)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2, classes=2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, img_rgb, img_depth, img_ir):
        missing_index = self.modality_drop(self.p, img_rgb.shape[0])
        
        x_rgb = self.special_bone_rgb(img_rgb*missing_index[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x_ir = self.special_bone_ir(img_ir*missing_index[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x_depth = self.special_bone_depth(img_depth*missing_index[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        evidence_miss, evidence_full = self.infer(x_rgb, x_ir, x_depth)
        alpha = dict()
        for v_num in range(3):
            alpha[v_num] = evidence_miss[v_num] + 1
        alpha_f = evidence_full+1
        alpha_a = self.DS_Combin((self.DS_Combin(alpha),alpha_f))
        evidence_a = alpha_a - 1

        pred_miss = evidence_a.data

        return pred_miss, alpha, alpha_f, alpha_a
        




class SURF_MD(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))

        self.net_rgb = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))        
        self.net_rgb_fc = model_resnet18_se_1.fc
        self.net_rgb_var = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))

        self.net_ir = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))       
        self.net_ir_fc = model_resnet18_se_2.fc
        self.net_ir_var = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))

        self.net_depth = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))               
        self.net_depth_fc = model_resnet18_se_3.fc
        self.net_depth_var = nn.Sequential(nn.Linear(512, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 1))
        
        
        self.shared_bone = nn.Sequential(nn.Linear(512*3, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(512, 2))
        
    

    def modality_drop(self, p, batchsize):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * batchsize]

            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()


        return p.squeeze()

    def infer(self, rgb, ir, depth): 

        rgb_logvar = self.net_rgb(rgb)
        rgb_logvar = torch.sigmoid(rgb_logvar)
        rgb = rgb * rgb_logvar

        depth_logvar = self.net_depth(depth)
        depth_logvar = torch.sigmoid(depth_logvar)
        depth = depth * depth_logvar

        ir_logvar = self.net_ir(ir)
        ir_logvar = torch.sigmoid(ir_logvar)
        ir = ir * ir_logvar
        
        return rgb, rgb_logvar, depth, depth_logvar, ir, ir_logvar


    
    def forward(self, img_rgb, img_depth, img_ir):
        missing_index = self.modality_drop(self.p, img_rgb.shape[0])
        
        x_rgb = self.special_bone_rgb(img_rgb*missing_index[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x_ir = self.special_bone_ir(img_ir*missing_index[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x_depth = self.special_bone_depth(img_depth*missing_index[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        rgb, rgb_logvar, depth, depth_logvar, ir, ir_logvar = self.infer(x_rgb, x_ir, x_depth)
        rgb_var = self.net_rgb_var(rgb)
        depth_var = self.net_depth_var(depth)
        ir_var = self.net_ir_var(ir)
        rgb = rgb_var * rgb * missing_index[:,0].unsqueeze(1)
        ir = ir_var * ir * missing_index[:,2].unsqueeze(1)
        depth = depth_var * depth * missing_index[:,1].unsqueeze(1)
        out = self.shared_bone(torch.cat((rgb,depth,ir),dim=-1))



        rgb_full, rgb_logvar, depth_full, depth_logvar, ir_full, ir_logvar = self.infer(self.special_bone_rgb(img_rgb), self.special_bone_ir(img_ir), self.special_bone_depth(img_depth)) 
        rgb_out = self.net_rgb_fc(rgb_full)
        depth_out = self.net_depth_fc(depth_full)
        ir_out = self.net_ir_fc(ir_full)
        rgb_var_full = self.net_rgb_var(rgb_full)
        depth_var_full = self.net_depth_var(depth_full)
        ir_var_full = self.net_ir_var(ir_full)

        return out, rgb_out, depth_out, ir_out, rgb_var_full, depth_var_full, ir_var_full, rgb_logvar, depth_logvar, ir_logvar
        




class SURF_UNCLLateNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.avg = False

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer,
                                              model_resnet18_se_1.layer3,
                                              model_resnet18_se_1.layer4,
                                              model_resnet18_se_1.avgpool,
                                              Flatten(1))
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer,
                                             model_resnet18_se_2.layer3,
                                             model_resnet18_se_2.layer4,
                                             model_resnet18_se_2.avgpool,
                                             Flatten(1))
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer,
                                                model_resnet18_se_3.layer3,
                                                model_resnet18_se_3.layer4,
                                                model_resnet18_se_3.avgpool,
                                                Flatten(1))

        
        self.net_rgb = nn.Sequential(nn.Linear(512, 1024),
                                    #  nn.ReLU(inplace=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Dropout(0.4),
                                     nn.Linear(1024, 512))
        
        self.net_ir =  nn.Sequential(nn.Linear(512, 1024),
                                    #  nn.ReLU(inplace=True),
                                    nn.LeakyReLU(0.1),
                                     nn.Dropout(0.4),
                                     nn.Linear(1024, 512))
        
        self.net_depth = nn.Sequential(nn.Linear(512, 1024),
                                       nn.LeakyReLU(0.1),
                                    #    nn.ReLU(inplace=True),
                                       nn.Dropout(0.4),
                                       nn.Linear(1024, 512))
                
        self.net_shared = nn.Sequential(nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.4),
                                        nn.Linear(1024, 512)) 
                
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 =  model_resnet18_se_1.fc
        
    def reparametrize(self, mu, logvar, k=1):
        eps = torch.randn((mu.size(0), k, mu.size(1)), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(mu.unsqueeze(1)).squeeze()
        return samples
    

    def Adaptive_Weight(self, mu, logvar, missing_index, eps=1e-12): 
        # logvar_clone = logvar.clone().detach().requires_grad_(True)
        # var = torch.exp(logvar_clone) + eps

        var = torch.exp(logvar) + eps

        T = 1. / var 
        # T = T.sum(-1).unsqueeze(2)
        missing_index = missing_index.squeeze()
        pd_mu = (mu * T * missing_index.unsqueeze(2)) / torch.sum(T * missing_index.unsqueeze(2), dim=1).unsqueeze(1)
        # pd_mu = (mu * missing_index.unsqueeze(2)) / torch.sum( missing_index.unsqueeze(2), dim=1).unsqueeze(1)

        return pd_mu
    

    def infer(self, rgb, ir, depth, p, prob): 
        rgb_logvar = self.net_rgb(rgb)
        mu     = rgb.unsqueeze(1)
        logvar = rgb_logvar.unsqueeze(1)

        depth_logvar = self.net_depth(depth)
        mu     = torch.cat((mu, depth.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, depth_logvar.unsqueeze(1)), dim=1)

        ir_logvar = self.net_ir(ir)
        mu     = torch.cat((mu, ir.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, ir_logvar.unsqueeze(1)), dim=1)
        if self.drop_mode == 'average':
            missing_index =  self.modality_drop(p, rgb_logvar.shape[0])
        else:
            missing_index =  self.unbalance_modality_drop(p, rgb_logvar.shape[0], prob)

        mu = self.Adaptive_Weight(mu, logvar, missing_index)
        mu_full = self.Adaptive_Weight(mu, logvar, torch.ones(missing_index.size()).cuda())
        
        return mu, rgb_logvar, depth_logvar, ir_logvar, missing_index


    def modality_drop(self, p, batchsize):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * batchsize]

            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()

        return p



    def unbalance_modality_drop(self, p, batchsize, prob):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]
        mode_num = 7

        if p == [0, 0, 0]:
            p = []
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            p = [p * batchsize]
            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
    
        p = p.float().cuda()

        return p



    def forward(self, img_rgb, img_depth, img_ir, prob=None, sample=False):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth) 
            
        miss_feat, rgb_logvar_full, depth_logvar_full, ir_logvar_full, missing_index = self.infer(x_rgb, x_ir, x_depth, self.p, prob)
        

        if self.avg == True:
            p = self.modality_drop(self.p, x_depth.shape[0])
            x_rgb_miss = x_rgb * p[:, 0]
            x_depth_miss = x_depth * p[:, 1]
            x_ir_miss = x_ir * p[:, 2]

            x = torch.cat((x_rgb_miss/3, x_depth_miss/3, x_ir_miss/3), dim=1)
            # x_full = torch.cat((x_rgb/3, x_depth/3, x_ir/3), dim=1)
        else:     
            
            x = torch.cat((miss_feat[:,0,:], miss_feat[:,1,:], miss_feat[:,2,:]), dim=1)
            # x_full = torch.cat((full_feat[:,0,:], full_feat[:,1,:], full_feat[:,2,:]), dim=1)

        layer_student = self.fc1(x)

        pd_sigma = self.net_shared(layer_student)

        if sample:
            x = self.reparametrize(layer_student, pd_sigma)
            miss_out = self.fc2(x)
        else:
            # miss_out = self.drop(layer_student)
            miss_out = self.fc2(layer_student)


        return miss_out, layer_student, x_rgb, rgb_logvar_full, x_ir, ir_logvar_full, x_depth, depth_logvar_full, pd_sigma, missing_index
        



# class SURF_UNCLLateNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         args.inplace_new = 384
#         model_resnet18_se_1 = resnet18_se(args, pretrained=False)
#         model_resnet18_se_2 = resnet18_se(args, pretrained=False)
#         model_resnet18_se_3 = resnet18_se(args, pretrained=False)
#         self.p = args.p
#         self.drop_mode = args.drop_mode
#         self.args = args
#         self.avg = False

#         self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
#                                               model_resnet18_se_1.bn1,
#                                               model_resnet18_se_1.relu,
#                                               model_resnet18_se_1.maxpool,
#                                               model_resnet18_se_1.layer1,
#                                               model_resnet18_se_1.layer2,
#                                               model_resnet18_se_1.se_layer,
#                                               model_resnet18_se_1.layer3,
#                                               model_resnet18_se_1.layer4,
#                                               model_resnet18_se_1.avgpool,
#                                               Flatten(1))
#         self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
#                                              model_resnet18_se_2.bn1,
#                                              model_resnet18_se_2.relu,
#                                              model_resnet18_se_2.maxpool,
#                                              model_resnet18_se_2.layer1,
#                                              model_resnet18_se_2.layer2,
#                                              model_resnet18_se_2.se_layer,
#                                              model_resnet18_se_2.layer3,
#                                              model_resnet18_se_2.layer4,
#                                              model_resnet18_se_2.avgpool,
#                                              Flatten(1))
#         self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
#                                                 model_resnet18_se_3.bn1,
#                                                 model_resnet18_se_3.relu,
#                                                 model_resnet18_se_3.maxpool,
#                                                 model_resnet18_se_3.layer1,
#                                                 model_resnet18_se_3.layer2,
#                                                 model_resnet18_se_3.se_layer,
#                                                 model_resnet18_se_3.layer3,
#                                                 model_resnet18_se_3.layer4,
#                                                 model_resnet18_se_3.avgpool,
#                                                 Flatten(1))

        
#         self.net_rgb = nn.Sequential(nn.Linear(512, 1024),
#                                     #  nn.ReLU(inplace=True),
#                                      nn.LeakyReLU(0.1),
#                                     #  nn.Dropout(0.5),
#                                      nn.Linear(1024, 512))
        
#         self.net_ir =  nn.Sequential(nn.Linear(512, 1024),
#                                     #  nn.ReLU(inplace=True),
#                                     nn.LeakyReLU(0.1),
#                                     #  nn.Dropout(0.5),
#                                      nn.Linear(1024, 512))
        
#         self.net_depth = nn.Sequential(nn.Linear(512, 1024),
#                                        nn.LeakyReLU(0.1),
#                                     #    nn.ReLU(inplace=True),
#                                     #    nn.Dropout(0.5),
#                                        nn.Linear(1024, 512))

                
#         self.fc1 = nn.Linear(512, 512)
#         self.fc2 =  model_resnet18_se_1.fc
        
#     def reparametrize(self, mu, logvar, k=1):
#         eps = torch.randn((mu.size(0), k, mu.size(1)), dtype=mu.dtype, device=mu.device)
#         samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(mu.unsqueeze(1)).squeeze()
#         return samples
    

#     def Adaptive_Weight(self, mu, logvar, missing_index, eps=1e-12): 
#         # logvar_clone = logvar.clone().detach().requires_grad_(True)
#         # var = torch.exp(logvar_clone) + eps
#         var = torch.exp(logvar) + eps

#         T = 1. / var 
#         missing_index = missing_index.squeeze()
#         pd_mu = (mu * T * missing_index.unsqueeze(2)) / torch.sum(T * missing_index.unsqueeze(2), dim=1).unsqueeze(1)
#         pd_sigma = torch.log(1 / torch.sum(T * missing_index.unsqueeze(2), dim=1)+eps)
        
#         return pd_mu, pd_sigma
    
    

#     def infer(self, rgb, ir, depth, p, prob): 
#         rgb_logvar = self.net_rgb(rgb)
#         mu     = rgb.unsqueeze(1)
#         logvar = rgb_logvar.unsqueeze(1)

#         depth_logvar = self.net_depth(depth)
#         mu     = torch.cat((mu, depth.unsqueeze(1)), dim=1)
#         logvar = torch.cat((logvar, depth_logvar.unsqueeze(1)), dim=1)

#         ir_logvar = self.net_ir(ir)
#         mu     = torch.cat((mu, ir.unsqueeze(1)), dim=1)
#         logvar = torch.cat((logvar, ir_logvar.unsqueeze(1)), dim=1)
#         if self.drop_mode == 'average':
#             missing_index =  self.modality_drop(p, rgb_logvar.shape[0])
#         else:
#             missing_index =  self.unbalance_modality_drop(p, rgb_logvar.shape[0], prob)

#         mu, pd_sigma = self.Adaptive_Weight(mu, logvar, missing_index)
#         mu_full, _ = self.Adaptive_Weight(mu, logvar, torch.ones(missing_index.size()).cuda())
        
#         return mu, rgb_logvar, depth_logvar, ir_logvar, missing_index, pd_sigma 


#     def modality_drop(self, p, batchsize):
#         modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
#         index_list = [x for x in range(7)]

#         if p == [0, 0, 0]:
#             p = []

#             prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
#             for i in range(batchsize):
#                 index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
#                 p.append(modality_combination[index])

#             p = np.array(p)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)

#         else:
#             p = p
#             # print(p)
#             p = [p * batchsize]

#             p = np.array(p).reshape(batchsize, 3)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)


#             # print(p[:, 0], p[:, 1], p[:, 2])
#         p = p.float().cuda()

#         return p



#     def unbalance_modality_drop(self, p, batchsize, prob):
#         modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
#         index_list = [x for x in range(7)]
#         mode_num = 7

#         if p == [0, 0, 0]:
#             p = []
#             for i in range(batchsize):
#                 index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
#                 p.append(modality_combination[index])

#             p = np.array(p)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)

#         else:
#             p = p
#             p = [p * batchsize]
#             p = np.array(p).reshape(batchsize, 3)
#             p = torch.from_numpy(p)
#             p = torch.unsqueeze(p, 2)
    
#         p = p.float().cuda()

#         return p



#     def forward(self, img_rgb, img_depth, img_ir, prob=None):
#         x_rgb = self.special_bone_rgb(img_rgb)
#         x_ir = self.special_bone_ir(img_ir)
#         x_depth = self.special_bone_depth(img_depth) 
            
#         miss_feat, rgb_logvar_full, depth_logvar_full, ir_logvar_full, missing_index, pd_sigma = self.infer(x_rgb, x_ir, x_depth, self.p, prob)

#         if self.avg == True:
#             p = self.modality_drop(self.p, x_depth.shape[0])
#             x_rgb_miss = x_rgb * p[:, 0]
#             x_depth_miss = x_depth * p[:, 1]
#             x_ir_miss = x_ir * p[:, 2]

#             x = x_rgb_miss/3 + x_depth_miss/3 + x_ir_miss/3
#             # x_full = torch.cat((x_rgb/3, x_depth/3, x_ir/3), dim=1)
#         else:     
            
#             x = miss_feat[:,0,:] + miss_feat[:,1,:] + miss_feat[:,2,:]


#         layer_student = self.fc1(x)

#         miss_out = self.fc2(layer_student)

#         # if torch.isnan((miss_out + rgb_out + ir_out + depth_out).sum()):
#         #     k = 1

#         return miss_out, layer_student, x_rgb, rgb_logvar_full, x_ir, ir_logvar_full, x_depth, depth_logvar_full, x, pd_sigma, missing_index
        




class SURF_Hemis(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)


        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        mu = (x_rgb + x_depth + x_ir)/p.sum(1)
        var = ((x_rgb-mu).pow(2)*p[:,0,:,:,:] + (x_depth-mu).pow(2)*p[:,1,:,:,:] + (x_ir-mu).pow(2)*p[:,2,:,:,:])/(p.sum(1)-1+1e-12)

        x = torch.cat((mu, var), dim=1)


        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4
    



class VarLayer(nn.Module):
    def __init__(self, r = 0, lora_alpha = 1, dim=512, hidden=1024):
        super(VarLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, hidden)
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, idx):
        x = self.fc1(x, idx)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x, idx)
        return x

class SURF_UNCLLateLoraNet(nn.Module):
    def __init__(self, args, r_ratio=0.1, lora_alpha=1):
        super().__init__()

        args.inplace_new = 384

        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.avg = False

        self.backbone = resnet18_se(args, pretrained=False, r_ratio=r_ratio, lora_alpha=lora_alpha)

        
        self.net_shared = VarLayer(int(512*r_ratio), lora_alpha, 512, 1024)
                
        self.net_fused = nn.Sequential(nn.Linear(512, 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.4),
                                        nn.Linear(1024, 512)) 
                
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 = nn.Linear(512, args.class_num)
        
    def reparametrize(self, mu, logvar, k=1):
        eps = torch.randn((mu.size(0), k, mu.size(1)), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(0.5*logvar.unsqueeze(1))).add_(mu.unsqueeze(1)).squeeze()
        return samples
    

    def Adaptive_Weight(self, mu, logvar, missing_index, eps=1e-12): 
        # logvar_clone = logvar.clone().detach().requires_grad_(True)
        var = torch.exp(logvar) + eps

        # var = (torch.exp(logvar) + eps).sum(-1)

        T = 1. / var 
        # T = T.unsqueeze(2)
        # T = T.sum(-1).unsqueeze(2)
        missing_index = missing_index.squeeze()
        pd_mu = (mu * T * missing_index.unsqueeze(2)) / torch.sum(T * missing_index.unsqueeze(2), dim=1).unsqueeze(1)
        # pd_mu = (mu * missing_index.unsqueeze(2)) / torch.sum( missing_index.unsqueeze(2), dim=1).unsqueeze(1)

        return pd_mu
    

    def infer(self, rgb, ir, depth, p, prob): 
        rgb_logvar = self.net_shared(rgb, 0)
        mu     = rgb.unsqueeze(1)
        logvar = rgb_logvar.unsqueeze(1)

        depth_logvar = self.net_shared(depth, 1)
        mu     = torch.cat((mu, depth.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, depth_logvar.unsqueeze(1)), dim=1)

        ir_logvar = self.net_shared(ir, 2)
        mu     = torch.cat((mu, ir.unsqueeze(1)), dim=1)
        logvar = torch.cat((logvar, ir_logvar.unsqueeze(1)), dim=1)
        if self.drop_mode == 'average':
            missing_index =  self.modality_drop(p, rgb_logvar.shape[0])
        else:
            missing_index =  self.unbalance_modality_drop(p, rgb_logvar.shape[0], prob)

        mu = self.Adaptive_Weight(mu, logvar, missing_index)
        mu_full = self.Adaptive_Weight(mu, logvar, torch.ones(missing_index.size()).cuda())
        
        return mu, rgb_logvar, depth_logvar, ir_logvar, missing_index


    def modality_drop(self, p, batchsize):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]

        if p == [0, 0, 0]:
            p = []

            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            # print(p)
            p = [p * batchsize]

            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)


            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()

        return p



    def unbalance_modality_drop(self, p, batchsize, prob):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]
        mode_num = 7

        if p == [0, 0, 0]:
            p = []
            for i in range(batchsize):
                index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[index])

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)

        else:
            p = p
            p = [p * batchsize]
            p = np.array(p).reshape(batchsize, 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
    
        p = p.float().cuda()

        return p



    def forward(self, img_rgb, img_depth, img_ir, prob=None, sample=False):
        x_rgb = self.backbone(img_rgb, 0)
        x_ir = self.backbone(img_ir, 2)
        x_depth = self.backbone(img_depth, 1) 
            
        miss_feat, rgb_logvar_full, depth_logvar_full, ir_logvar_full, missing_index = self.infer(x_rgb, x_ir, x_depth, self.p, prob)
        

        if self.avg == True:
            p = self.modality_drop(self.p, x_depth.shape[0])
            x_rgb_miss = x_rgb * p[:, 0]
            x_depth_miss = x_depth * p[:, 1]
            x_ir_miss = x_ir * p[:, 2]

            x = torch.cat((x_rgb_miss/3, x_depth_miss/3, x_ir_miss/3), dim=1)
            # x_full = torch.cat((x_rgb/3, x_depth/3, x_ir/3), dim=1)
        else:     
            
            x = torch.cat((miss_feat[:,0,:], miss_feat[:,1,:], miss_feat[:,2,:]), dim=1)
            # x_full = torch.cat((full_feat[:,0,:], full_feat[:,1,:], full_feat[:,2,:]), dim=1)

        layer_student = self.fc1(x)

        pd_sigma = self.net_fused(layer_student)

        if sample:
            x = self.reparametrize(layer_student, pd_sigma)
            miss_out = self.fc2(x)
        else:
            # miss_out = self.drop(layer_student)
            miss_out = self.fc2(layer_student)


        return miss_out, layer_student, x_rgb, rgb_logvar_full, x_ir, ir_logvar_full, x_depth, depth_logvar_full, pd_sigma, missing_index


class SURF_trmlora(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se_trm(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = resnet18_uni_trm(args, pretrained=False)
    
        self.special_bone_ir = resnet18_uni_trm(args, pretrained=False)
        
        self.special_bone_depth = resnet18_uni_trm(args, pretrained=False)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1)
                                         )
        trm_layer = TransformerDecoderLayer3lora(args.r_ratio, args.lora_alpha, 512, 8, batch_first=True)
        self.trm_lora = TransformerDecoder3lora(trm_layer, args.layer_num)
        self.fc = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.index = None

    def ftr_modal_drop(self,x_rgb, x_depth,x_ir):
        p = self.p
        self.index = self.index if self.index is not None else np.zeros(x_rgb.shape[0],dtype=np.int8)-1
        if p == [0, 0, 0]:
            modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            index_list = [x for x in range(7)]
            p = []
            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                if self.index[i]<0:
                    self.index[i] = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[self.index[i]])
            p = np.array(p)
            p = torch.from_numpy(p)
        else:
            p = p
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
        p = p.float().cuda()
        p =  torch.unsqueeze(p, 2)
        x_rgb = x_rgb * p[:, 0]
        # print(x_rgb)
        x_depth = x_depth * p[:, 1]
        # print(x_depth)
        x_ir = x_ir * p[:, 2]
        return x_rgb, x_depth,x_ir, torch.squeeze(p,2)
    
    def modality_drop(self, x_rgb, x_depth,x_ir, p):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]
        self.index = self.index if self.index is not None else np.zeros(x_rgb.shape[0],dtype=np.int8)-1

        if p == [0, 0, 0]:
            p = []

            # for i in range(x_rgb.shape[0]):
            #     index = random.randint(0, 6)
            #     p.append(modality_combination[index])
            #     if 'model_arch_index' in args.writer_dicts.keys():
            #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                if self.index[i] < 0:
                    self.index[i] = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[self.index[i]])
                # if 'model_arch_index' in args.writer_dicts.keys():
                #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
            p = torch.unsqueeze(p, 3)
            p = torch.unsqueeze(p, 4)

        else:
            # print("start:")
            p = p
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
            p = torch.unsqueeze(p, 3)
            p = torch.unsqueeze(p, 4)

            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()
        
        x_rgb = x_rgb * p[:, 0]
        # print(x_rgb)
        x_depth = x_depth * p[:, 1]
        # print(x_depth)
        x_ir = x_ir * p[:, 2]
        # print(x_ir)

        return x_rgb, x_depth,x_ir


    def forward(self, img_rgb, img_depth, img_ir):
        # print("img_shape:")
        # print(img_rgb.shape, img_depth.shape, img_ir.shape)
        self.index=None
        x_rgb, se_rgb = self.special_bone_rgb(img_rgb)
        x_depth, se_depth = self.special_bone_depth(img_depth)
        x_ir, se_ir = self.special_bone_ir(img_ir)
        # print("x_shape:")
        # print(x_rgb.shape, x_depth.shape, x_ir.shape)

        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir, _p = self.ftr_modal_drop(x_rgb, x_depth, x_ir)
            se_rgb, se_depth, se_ir = self.modality_drop(se_rgb, se_depth, se_ir, self.p)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)


        x = torch.cat((se_rgb, se_depth, se_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        shared_x = self.shared_bone[3](x)
        shared_x = torch.unsqueeze(shared_x,1)
        x_rgb = torch.unsqueeze(x_rgb, 1)
        x_depth = torch.unsqueeze(x_depth, 1)
        x_ir = torch.unsqueeze(x_ir, 1)
        x1 = self.trm_lora(0, shared_x, x_rgb)
        x2 = self.trm_lora(1, shared_x, x_depth)
        x3 = self.trm_lora(2, shared_x, x_ir)
        x1 = torch.squeeze(x1, 1)
        x2 = torch.squeeze(x2, 1)
        x3 = torch.squeeze(x3, 1)
        x1,x2,x3,_ = self.ftr_modal_drop(x1, x2, x3)
        x= (x1+x2+x3)/torch.sum(_p,dim=-1,keepdim=True)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SURF_01decomp(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se_trm(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = resnet18_uni_trm(args, pretrained=False)
    
        self.special_bone_ir = resnet18_uni_trm(args, pretrained=False)
        
        self.special_bone_depth = resnet18_uni_trm(args, pretrained=False)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1)
                                         )
        self.proj = nn.Linear(512, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512,2)

    def ftr_modal_drop(self,x_rgb, x_depth,x_ir):
        p = self.p
        self.index = self.index if self.index is not None else np.zeros(x_rgb.shape[0],dtype=np.int8)-1
        if p == [0, 0, 0]:
            modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            index_list = [x for x in range(7)]
            p = []
            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                if self.index[i]<0:
                    self.index[i] = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[self.index[i]])
            p = np.array(p)
            p = torch.from_numpy(p)
        else:
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
        p = p.float().cuda()
        p =  torch.unsqueeze(p, 2)
        x_rgb = x_rgb * p[:, 0]
        # print(x_rgb)
        x_depth = x_depth * p[:, 1]
        # print(x_depth)
        x_ir = x_ir * p[:, 2]
        return x_rgb, x_depth,x_ir
    
    def modality_drop(self, x_rgb, x_depth,x_ir, p):
        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        index_list = [x for x in range(7)]
        self.index = self.index if self.index is not None else np.zeros(x_rgb.shape[0],dtype=np.int8)-1

        if p == [0, 0, 0]:
            p = []

            # for i in range(x_rgb.shape[0]):
            #     index = random.randint(0, 6)
            #     p.append(modality_combination[index])
            #     if 'model_arch_index' in args.writer_dicts.keys():
            #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
            prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
            for i in range(x_rgb.shape[0]):
                if self.index[i] < 0:
                    self.index[i] = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
                p.append(modality_combination[self.index[i]])
                # if 'model_arch_index' in args.writer_dicts.keys():
                #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

            p = np.array(p)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
            p = torch.unsqueeze(p, 3)
            p = torch.unsqueeze(p, 4)

        else:
            # print("start:")
            p = p
            # print(p)
            p = [p * x_rgb.shape[0]]
            # print(p)
            p = np.array(p).reshape(x_rgb.shape[0], 3)
            p = torch.from_numpy(p)
            p = torch.unsqueeze(p, 2)
            p = torch.unsqueeze(p, 3)
            p = torch.unsqueeze(p, 4)

            # print(p[:, 0], p[:, 1], p[:, 2])
        p = p.float().cuda()
        
        x_rgb = x_rgb * p[:, 0]
        # print(x_rgb)
        x_depth = x_depth * p[:, 1]
        # print(x_depth)
        x_ir = x_ir * p[:, 2]
        # print(x_ir)

        return x_rgb, x_depth,x_ir, p




    def forward(self, img_rgb, img_depth, img_ir):
        self.index = None
        # print("img_shape:")
        # print(img_rgb.shape, img_depth.shape, img_ir.shape)
        x_rgb, se_rgb = self.special_bone_rgb(img_rgb)
        x_depth, se_depth = self.special_bone_depth(img_depth)
        x_ir, se_ir = self.special_bone_ir(img_ir)
        # print("x_shape:")
        # print(x_rgb.shape, x_depth.shape, x_ir.shape)

        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir = self.ftr_modal_drop(x_rgb, x_depth, x_ir)
            se_rgb, se_depth, se_ir, _ = self.modality_drop(se_rgb, se_depth, se_ir, self.p)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        # print(se_rgb.shape)
        # print(x_rgb.shape)

        x = torch.cat((se_rgb, se_depth, se_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        shared_x = self.shared_bone[3](x)
        b_rgb = self.proj(x_rgb)
        b_depth = self.proj(x_depth)
        b_ir = self.proj(x_ir)
        b_shared = self.proj(shared_x)
        b_fused = torch.sigmoid(3*(b_depth+b_ir+b_rgb-self.args.gate01))
        x1 = self.dropout(b_fused)
        x1= self.fc(x1)
        x2 = self.dropout(b_shared)
        x2= self.fc(x2)
        # print(x.shape)
        return (x1+x2)/2,b_rgb,b_depth,b_ir,b_shared,b_fused