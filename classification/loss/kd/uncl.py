from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist



class PGCL(nn.Module):
    def __init__(self):
        super(PGCL, self).__init__()

    def forward(self, fm_rgb, fm_depth, fm_ir, fm_t):

        norm_fm_rgb = F.normalize(fm_rgb, p=2, dim=-1)
        G_rgb = 1 - 0.5 * (norm_fm_rgb.unsqueeze(1) - norm_fm_rgb).pow(2).sum(-1)

        norm_fm_ir = F.normalize(fm_ir, p=2, dim=-1)
        G_ir = 1 - 0.5 * (norm_fm_ir.unsqueeze(1) - norm_fm_ir).pow(2).sum(-1)

        norm_fm_depth = F.normalize(fm_depth, p=2, dim=-1)
        G_depth = 1 - 0.5 * (norm_fm_depth.unsqueeze(1) - norm_fm_depth).pow(2).sum(-1)

        norm_fm_t = F.normalize(fm_t, p=2, dim=-1)
        G_t = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        loss_rgb = (G_t/0.1 - G_rgb/0.1).pow(2).mean(-1) 
        loss_depth = (G_t/0.1 - G_depth/0.1).pow(2).mean(-1) 
        loss_ir = (G_t/0.1 - G_ir/0.1).pow(2).mean(-1) 
        loss = loss_rgb.mean() + loss_depth.mean() + loss_ir.mean()

        return loss 


class CL(nn.Module):
    def __init__(self):
        super(CL, self).__init__()

    def forward(self, fm_s, fm_t, targets, fusion_true, multi_index=None):
        if fusion_true:
            mask = targets[multi_index].unsqueeze(1) - targets
            self_mask_pos = 1 - (torch.zeros_like(mask) != mask).float()            
        else:
            mask = targets.unsqueeze(1) - targets
            self_mask_pos = 1 - (torch.zeros_like(mask) != mask).float()
    
        loss_pos = (((fm_s.unsqueeze(1) - fm_t).pow(2).sum(-1)) * self_mask_pos)/self_mask_pos.sum(-1).unsqueeze(1)

        # loss = F.relu(((fm_s - fm_t).pow(2) / (2 * logvar_s.exp() + 1e-12) + 0.5 * logvar_s)).sum(-1).mean()
        if fusion_true:
            return 7 * loss_pos.sum(-1).sum()/fm_t.size(0) 
            # return loss_pos.sum(-1).mean() + loss_neg.sum(-1).mean()
        else:
            return loss_pos.sum(-1).mean()




class UNCL(nn.Module):
    def __init__(self):
        super(UNCL, self).__init__()

    def forward(self, fm_s, fm_t, logvar_s, targets, fusion_true, multi_index=None):

        if fusion_true:
            mask = targets[multi_index].unsqueeze(1) - targets
            self_mask_pos = 1 - (torch.zeros_like(mask) != mask).float()            
        else:
            mask = targets.unsqueeze(1) - targets
            self_mask_pos = 1 - (torch.zeros_like(mask) != mask).float()
    
        loss_pos = ((F.relu(((fm_s.unsqueeze(1) - fm_t).pow(2) / (2 * logvar_s.exp().unsqueeze(1) + 1e-12)) + 0.5 * logvar_s.unsqueeze(1)).sum(-1)) * self_mask_pos)/self_mask_pos.sum(-1).unsqueeze(1)#.diag()#
        # loss_pos = ((torch.clip(((fm_s.unsqueeze(1) - fm_t).pow(2) / (2 * logvar_s.exp().unsqueeze(1) + 1e-12)) + 0.5 * logvar_s.unsqueeze(1),0,torch.tensor(100).cuda()).sum(-1)) * self_mask_pos)/self_mask_pos.sum(-1).unsqueeze(1)#.diag()#
        loss_neg = ((F.relu(-((fm_s.unsqueeze(1) - fm_t).pow(2) / (2 * logvar_s.exp().unsqueeze(1) + 1e-12)) - 0.5 * logvar_s.unsqueeze(1)).sum(-1)) * (1 - self_mask_pos))/(1-self_mask_pos).sum(-1).unsqueeze(1)
        # loss = F.relu(((fm_s - fm_t).pow(2) / (2 * logvar_s.exp() + 1e-12) + 0.5 * logvar_s)).sum(-1).mean()
        if fusion_true:
            return 7 * loss_pos.sum(-1).sum()/fm_t.size(0) + 7 * loss_neg.sum(-1).sum()/fm_t.size(0)
            # return loss_pos.sum(-1).mean() + loss_neg.sum(-1).mean()
        else:
            return loss_pos.sum(-1).mean() + loss_neg.sum(-1).mean()
    

# class GCL(nn.Module):
#     def __init__(self):
#         super(GCL, self).__init__()

#     def forward(self, fm_rgb, fm_depth, fm_ir, fm_fusion, multi_index, fm_t, weight):
                                                                                                                        
#         norm_fm_rgb = F.normalize(fm_rgb, p=2, dim=-1)
#         G_rgb = 1 - 0.5 * (norm_fm_rgb.unsqueeze(1) - norm_fm_rgb).pow(2).sum(-1)

#         norm_fm_ir = F.normalize(fm_ir, p=2, dim=-1)
#         G_ir = 1 - 0.5 * (norm_fm_ir.unsqueeze(1) - norm_fm_ir).pow(2).sum(-1)

#         norm_fm_depth = F.normalize(fm_depth, p=2, dim=-1)
#         G_depth = 1 - 0.5 * (norm_fm_depth.unsqueeze(1) - norm_fm_depth).pow(2).sum(-1)

#         norm_fm_fusion = F.normalize(fm_fusion, p=2, dim=-1)
#         G_fusion = 1 - 0.5 * (norm_fm_fusion.unsqueeze(1) - norm_fm_fusion).pow(2).sum(-1)

#         norm_fm_t = F.normalize(fm_t, p=2, dim=-1)
#         G_t = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

       
#         loss_rgb = (G_t/0.1 - G_rgb/0.1).pow(2).mean(-1)# * weight[:, 0]
#         loss_depth = (G_t/0.1 - G_depth/0.1).pow(2).mean(-1)# * weight[:, 1]
#         loss_ir = (G_t/0.1 - G_ir/0.1).pow(2).mean(-1)# * weight[:, 2]
#         loss_fusion = (G_t/0.1 - G_fusion/0.1).pow(2).mean(-1)# * weight[:, 3]
#         loss =  loss_rgb.mean() + loss_depth.mean() + loss_ir.mean()# + loss_fusion.mean()

#         return loss 
    


# class GCL(nn.Module):
#     def __init__(self):
#         super(GCL, self).__init__()

#     def softmax(self, f, mask_neg):
#         exp_f = (f/0.1).exp()
#         exp_neg = exp_f * mask_neg
#         exp_pos = exp_f * (1 - mask_neg)

#         p = exp_pos/(exp_neg.sum(-1).unsqueeze(1) + exp_pos)
#         return p
    

#     def forward(self, fm_rgb, fm_depth, fm_ir, fm_fusion, multi_index, fm_t, targets):

        # norm_fm_t = F.normalize(fm_t, p=2, dim=-1)
        # G_t = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        # norm_fm_rgb = F.normalize(fm_rgb, p=2, dim=-1)
        # G_rgb = 1 - 0.5 * (norm_fm_rgb.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        # norm_fm_ir = F.normalize(fm_ir, p=2, dim=-1)
        # G_ir = 1 - 0.5 * (norm_fm_ir.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        # norm_fm_depth = F.normalize(fm_depth, p=2, dim=-1)
        # G_depth = 1 - 0.5 * (norm_fm_depth.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        # norm_fm_fusion = F.normalize(fm_fusion, p=2, dim=-1)
        # G_fusion = 1 - 0.5 * (norm_fm_fusion.unsqueeze(1) - norm_fm_t).pow(2).sum(-1)

        # p_rgb = F.softmax(-((G_t.unsqueeze(1) - G_rgb).pow(2).mean(-1)), dim=-1)
        # p_depth = F.softmax(-((G_t.unsqueeze(1) - G_depth).pow(2).mean(-1)), dim=-1)
        # p_ir = F.softmax(-((G_t.unsqueeze(1) - G_ir).pow(2).mean(-1)), dim=-1)
        # p_fusion = F.softmax(-((G_t.unsqueeze(1) - G_fusion).pow(2).mean(-1)), dim=-1)

        # G_t = (fm_t.unsqueeze(1) - fm_t).pow(2).sum(-1)
        # G_rgb = (fm_rgb.unsqueeze(1) - fm_rgb).pow(2).sum(-1)
        # G_ir = (fm_ir.unsqueeze(1) - fm_ir).pow(2).sum(-1)
        # G_depth = (fm_depth.unsqueeze(1) - fm_depth).pow(2).sum(-1)
        # # G_fusion = (fm_fusion.unsqueeze(1) - fm_fusion).pow(2).sum(-1)
        # G_fusion = (fm_fusion[multi_index].unsqueeze(1) - fm_fusion[multi_index]).pow(2).sum(-1)

        # norm_fm_t = F.normalize(G_t, p=2, dim=-1)

        # norm_fm_rgb = F.normalize(G_rgb, p=2, dim=-1)
        # p_rgb = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_rgb).pow(2).sum(-1)

        # norm_fm_ir = F.normalize(G_ir, p=2, dim=-1)
        # p_ir = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_ir).pow(2).sum(-1)

        # norm_fm_depth = F.normalize(G_depth, p=2, dim=-1)
        # p_depth = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_depth).pow(2).sum(-1)

        # G_t_fusion = (fm_t[multi_index].unsqueeze(1) - fm_t[multi_index]).pow(2).sum(-1)
        # # G_t_fusion = (fm_t.unsqueeze(1) - fm_t).pow(2).sum(-1)
        # norm_fm_t_fusion = F.normalize(G_t_fusion, p=2, dim=-1)
        # norm_fm_fusion = F.normalize(G_fusion, p=2, dim=-1)
        # p_fusion = 1 - 0.5 * (norm_fm_t_fusion.unsqueeze(1) - norm_fm_fusion).pow(2).sum(-1)


        # mask = targets.unsqueeze(1) - targets
        # self_mask = (torch.zeros_like(mask) != mask).float()
        
        # # p_rgb = F.softmax(p_rgb, dim=-1)
        # # p_depth = F.softmax(p_depth, dim=-1)
        # # p_ir = F.softmax(p_ir, dim=-1)
        # # p_fusion = F.softmax(p_fusion, dim=-1)

        # p_rgb = self.softmax(p_rgb, self_mask)
        # p_depth = self.softmax(p_depth, self_mask)
        # p_ir = self.softmax(p_ir, self_mask)
        # mask_fusion = targets[multi_index].unsqueeze(1) - targets[multi_index]
        # # mask_fusion = targets.unsqueeze(1) - targets
        # self_mask_fusion = (torch.zeros_like(mask_fusion) != mask_fusion).float()
        # p_fusion = self.softmax(p_fusion, self_mask_fusion)


        # loss_rgb = - torch.log(p_rgb + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        # loss_depth = - torch.log(p_depth + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        # loss_ir = - torch.log(p_ir + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        # loss_fusion = - torch.log(p_fusion + 1e-12) * (1 - self_mask_fusion)/(1 - self_mask_fusion).sum(-1).unsqueeze(1)#.diag()#
        # # loss_fusion = - torch.log(p_fusion + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1) 


        # loss = loss_rgb.sum(-1).mean() + loss_depth.sum(-1).mean() + loss_ir.sum(-1).mean() + loss_fusion.sum(-1).mean()#7 * loss_fusion.sum(-1).sum()/fm_t.size(0) # 
        # # loss = loss_rgb.mean() + loss_depth.mean() + loss_ir.mean() + loss_fusion.mean()

        # return loss 
    

class GCL(nn.Module):
    def __init__(self, T=0.1):
        super(GCL, self).__init__()
        self.T = T
        # self.T = nn.Parameter(torch.ones([]) * T)

    def softmax(self, f, mask_neg):
        exp_f = (f/self.T).exp()
        exp_neg = exp_f * mask_neg
        exp_pos = exp_f * (1 - mask_neg)

        p = exp_pos/(exp_neg.sum(-1).unsqueeze(1) + exp_pos)
        return p
    

    def forward(self, fm_rgb, fm_depth, fm_ir, fm_fusion, multi_index, fm_t, targets):

        G_t = (fm_t.unsqueeze(1) - fm_t.detach()).pow(2).sum(-1)
        G_rgb = (fm_rgb.unsqueeze(1) - fm_rgb.detach()).pow(2).sum(-1)
        G_ir = (fm_ir.unsqueeze(1) - fm_ir.detach()).pow(2).sum(-1)
        G_depth = (fm_depth.unsqueeze(1) - fm_depth.detach()).pow(2).sum(-1)
        # G_fusion = (fm_fusion.unsqueeze(1) - fm_fusion).pow(2).sum(-1)
        G_fusion = (fm_fusion[multi_index].unsqueeze(1) - fm_fusion.detach()).pow(2).sum(-1)

        norm_fm_t = F.normalize(G_t, p=2, dim=-1)

        norm_fm_rgb = F.normalize(G_rgb, p=2, dim=-1)
        p_rgb = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_rgb).pow(2).sum(-1)

        norm_fm_ir = F.normalize(G_ir, p=2, dim=-1)
        p_ir = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_ir).pow(2).sum(-1)

        norm_fm_depth = F.normalize(G_depth, p=2, dim=-1)
        p_depth = 1 - 0.5 * (norm_fm_t.unsqueeze(1) - norm_fm_depth).pow(2).sum(-1)

        # G_t_fusion = (fm_t[multi_index].unsqueeze(1) - fm_t).pow(2).sum(-1)
        G_t_fusion = (fm_t.unsqueeze(1) - fm_t).pow(2).sum(-1)
        norm_fm_t_fusion = F.normalize(G_t_fusion, p=2, dim=-1)
        norm_fm_fusion = F.normalize(G_fusion, p=2, dim=-1)
        p_fusion = 1 - 0.5 * (norm_fm_t_fusion.unsqueeze(1) - norm_fm_fusion).pow(2).sum(-1)


        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask).float()
        
        # p_rgb = F.softmax(p_rgb, dim=-1)
        # p_depth = F.softmax(p_depth, dim=-1)
        # p_ir = F.softmax(p_ir, dim=-1)
        # p_fusion = F.softmax(p_fusion, dim=-1)

        p_rgb = self.softmax(p_rgb, self_mask)
        p_depth = self.softmax(p_depth, self_mask)
        p_ir = self.softmax(p_ir, self_mask)
        mask_fusion = targets.unsqueeze(1) - targets[multi_index]
        # mask_fusion = targets.unsqueeze(1) - targets
        self_mask_fusion = (torch.zeros_like(mask_fusion) != mask_fusion).float()
        p_fusion = self.softmax(p_fusion, self_mask_fusion)


        loss_rgb = - torch.log(p_rgb + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        loss_depth = - torch.log(p_depth + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        loss_ir = - torch.log(p_ir + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1)#.diag()#
        loss_fusion = - torch.log(p_fusion + 1e-12) * (1 - self_mask_fusion)/((1 - self_mask_fusion).sum(-1)+1e-12).unsqueeze(1)#.diag()#
        # loss_fusion = - torch.log(p_fusion + 1e-12) * (1 - self_mask)/(1 - self_mask).sum(-1).unsqueeze(1) 


        loss = loss_rgb.sum(-1).mean() + loss_depth.sum(-1).mean() + loss_ir.sum(-1).mean() + 7 * loss_fusion.sum(-1).sum()/fm_t.size(0) # loss_fusion.sum(-1).mean()#
        # loss = loss_rgb.mean() + loss_depth.mean() + loss_ir.mean() + loss_fusion.mean()

        return loss 