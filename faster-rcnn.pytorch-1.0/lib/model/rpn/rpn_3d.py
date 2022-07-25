from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config_3d import cfg
from .proposal_layer_3d import _ProposalLayer
from .anchor_target_layer_3d import _AnchorTargetLayer
from model.utils.net_utils_3d import _smooth_l1_loss, compute_diou, FocalLoss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES # [8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS # [0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0] # [16, ]

        # define the convrelu layers processing input feature map
        # 所有的conv层都是：kernel_size=3，pad=1，stride=1
        # 所有的pooling层都是：kernel_size=2，pad=0，stride=2
        self.RPN_Conv = nn.Conv3d(self.din, 512, kernel_size=3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 6 # 6(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv3d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    # reshape 便于softmax分类
    # 图片的shape是（B, D=18, W, H, Depth) : 2(bg/fg) * 9 (anchors)
    # 进行softmax的matrix的一边需要等于num of class，在这里是一个二分类，即是否含有物体，所以是2）。
    # 所以我们会把（W，H, Depth，D=18)reshape成（B,2，9*W*H, Depth）
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        #print(f"reshape input shape:{input_shape}")
        x = x.view(
            input_shape[0], # batch size
            int(d), # dim
            int(float(input_shape[1] * input_shape[2] ) / float(d)), # 9*W
            input_shape[3], # H
            input_shape[4] # Depth
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)


        # get rpn classification score
        #print(f"RPN class reshape progress")
        # (1, 18, 8, 8, 8)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # print(f"rpn_cls_score shape:{rpn_cls_score.shape}")
        # (1, 2, 576, 8)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        #print(f"rpn_cls_score_reshape shape:{rpn_cls_score_reshape.shape}") # [b, 2, 8*9, 8, 8]
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        #print(f"rpn_cls_prob(經過softmax) shape:{rpn_cls_prob_reshape.shape}")
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        #print(f'rpn_cls_prob shape:{rpn_cls_prob.shape}') # [b, 2*9, 8, 8. 8]

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        #  [b, post_nms_topN, 7]
        # 可以判斷rois的位置
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))
        # print(f'rois shape:{rois.shape}') [b, 2000, 7] post_nms_topN
        #print(f'Max rois:{rois[:,0,:]}')

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            # print(f'RPN class score:{rpn_cls_score.shape}') # 有負值 要再確認一下
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # print(f'rpn data:{len(rpn_data)}')
            #print('===== rpn_loss progress =====')
            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2)
          
            # 拿到class label
            rpn_label = rpn_data[0].view(batch_size, -1)
            # print(f'rpn_label shape:{rpn_label.shape}') # [4, N]
            # print(f'rpn_label:{rpn_label}')
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            # print(f'rpn_keep shape:{rpn_keep.shape}')
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            # Cross entropy
            if cfg.TRAIN.CLASS_LOSS == 'cross_entropy':
                self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # Focal loss
            elif cfg.TRAIN.CLASS_LOSS == 'focal_loss':
                focal_loss = FocalLoss(class_num=2, alpha=0.25, gamma=2)
                self.rpn_loss_cls = focal_loss(rpn_cls_score, rpn_label)
            else:
                print('Classification loss only cross_entropy or focal_loss')
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            
            # rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights [b, 9*6, 8, 8 ,8]
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            #print(f'rpn_bbox_targets shape:{rpn_bbox_targets.shape}')
            #print(f'rpn_bbox_inside_weights shape:{rpn_bbox_inside_weights.shape}')
            #print(f'rpn_bbox_outside_weights shape:{rpn_bbox_outside_weights.shape}')

            # compute bbox regression loss
            # print(f'rpn_bbox_targets:{rpn_bbox_targets}') # 幾乎都是零 沒有target
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            
            #self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
            #                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
            # 
            # regression loss
            # L1 loss function 
            if cfg.TRAIN.BOX_REGRESSION_LOSS == 'L1':
                self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
            # DIOU loss function
            elif cfg.TRAIN.BOX_REGRESSION_LOSS == 'diou':
                _, self.rpn_loss_box = compute_diou(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                                rpn_bbox_outside_weights)    
            else:
                print('Regression loss only L1 or diou')
            
        return rois, self.rpn_loss_cls, self.rpn_loss_box
