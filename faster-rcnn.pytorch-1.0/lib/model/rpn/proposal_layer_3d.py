from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config_3d import cfg
from .generate_anchors_3d import generate_anchors
from .bbox_transform_3d import bbox_transform_inv, clip_boxes, clip_boxes_batch
# nms 要改
#from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        # anchor 還未加上偏移量
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
            ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        
        # print(f"anchor num must be 9, true:{self._num_anchors}")

        # rois blob: holds R regions of interest, each is a 7-tuple
        # (n, x1, y1, x2, y2, z1, z2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2, z1, z2)
        # top[0].reshape(1, 7)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    @staticmethod    
    def calc_iou(a, b): 
        # a,b (x1,y1,x2,y2,z1,z2)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) * (b[:, 5] - b[:, 4])
        #   area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1) area 不確定要不要+1

        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
        idp = torch.min(torch.unsqueeze(a[:, 5], dim=1), b[:, 5]) - torch.max(torch.unsqueeze(a[:, 4], 1), b[:, 4])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)
        idp = torch.clamp(idp, min=0) 

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) * (a[:, 5] - a[:, 4]), dim=1) + area - iw * ih *idp 

        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih * idp

        IoU = intersection / ua
        
        '''
        medicaldetectiontoolkit 的方法
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = iou.view(boxes2_repeat, boxes1_repeat)
        '''
        return IoU

    @staticmethod   
    def calc_diou(bboxes1, bboxes2):
        # 確認bboxes1和bboxes2維度
        # bboxes1 : (N,6)  bboxes2:(M,6)
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        dious = torch.zeros((rows, cols))
        if rows * cols == 0:#
            return dious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            dious = torch.zeros((cols, rows))
            exchange = True
        # xmin,ymin,xmax,ymax,zmin,zmax -> [:,0],[:,1],[:,2],[:,3],[:,4],[:,5]
        # (N,M)
        w1 = (bboxes1[:, 2] - bboxes1[:, 0]).unsqueeze(1).expand(rows,cols)
        h1 = (bboxes1[:, 3] - bboxes1[:, 1]).unsqueeze(1).expand(rows,cols) 
        d1 = (bboxes1[:, 5] - bboxes1[:, 4]).unsqueeze(1).expand(rows,cols)
        w2 = (bboxes2[:, 2] - bboxes2[:, 0]).unsqueeze(0).expand(rows,cols)
        h2 = (bboxes2[:, 3] - bboxes2[:, 1]).unsqueeze(0).expand(rows,cols)
        d2 = (bboxes2[:, 5] - bboxes2[:, 4]).unsqueeze(0).expand(rows,cols)
        
        area1 = w1 * h1 * d1 # (N,M)
        area2 = w2 * h2 * d2 # (N,M)
        
        # (N,M)
        center_x1 = ((bboxes1[:, 2] + bboxes1[:, 0]) / 2).expand(rows,cols) 
        center_y1 = ((bboxes1[:, 3] + bboxes1[:, 1]) / 2).expand(rows,cols) 
        center_z1 = ((bboxes1[:, 4] + bboxes1[:, 5]) / 2).expand(rows,cols) 
        center_x2 = ((bboxes2[:, 2] + bboxes2[:, 0]) / 2).expand(rows,cols) 
        center_y2 = ((bboxes2[:, 3] + bboxes2[:, 1]) / 2).expand(rows,cols) 
        center_z2 = ((bboxes2[:, 4] + bboxes2[:, 5]) / 2).expand(rows,cols) 
        
        # inter 
        # 避免bboxes1與bboxes2維度不同
        # iw, ih ,idp 維度為(N,M)
        iw = torch.min(torch.unsqueeze(bboxes1[:, 2], dim=1), bboxes2[:, 2]) - torch.max(torch.unsqueeze(bboxes1[:, 0], 1), bboxes2[:, 0])
        ih = torch.min(torch.unsqueeze(bboxes1[:, 3], dim=1), bboxes2[:, 3]) - torch.max(torch.unsqueeze(bboxes1[:, 1], 1), bboxes2[:, 1])
        idp = torch.min(torch.unsqueeze(bboxes1[:, 5], dim=1), bboxes2[:, 5]) - torch.max(torch.unsqueeze(bboxes1[:, 4], 1), bboxes2[:, 4])
        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)
        idp = torch.clamp(idp, min=0) 
        inter_area = iw * ih * idp # (N,M)
        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2 + (center_z2 - center_z1)**2
        
        # Outer
        # 避免bboxes1與bboxes2維度不同
        # ow, oh ,odp 維度為(N,M)
        ow = torch.max(torch.unsqueeze(bboxes1[:, 2], dim=1), bboxes2[:, 2]) - torch.min(torch.unsqueeze(bboxes1[:, 0], 1), bboxes2[:, 0])
        oh = torch.max(torch.unsqueeze(bboxes1[:, 3], dim=1), bboxes2[:, 3]) - torch.min(torch.unsqueeze(bboxes1[:, 1], 1), bboxes2[:, 1])
        odp = torch.max(torch.unsqueeze(bboxes1[:, 5], dim=1), bboxes2[:, 5]) - torch.min(torch.unsqueeze(bboxes1[:, 4], 1), bboxes2[:, 4])
        ow = torch.clamp(ow, min=0)
        oh = torch.clamp(oh, min=0)
        odp = torch.clamp(odp, min=0) 
        outer_diag = ow ** 2 + oh ** 2 + odp ** 2 + 1e-7
        
        union = area1 + area2 - inter_area + 1e-7
        dious = inter_area / union - (inter_diag) / outer_diag
        dious = torch.clamp(dious,min=-1.0,max = 1.0)
        if exchange:
            dious = dious.T
        return dious


    def Weighted_cluster_nms(self, boxes, scores, NMS_threshold=0.7):
        '''
        Arguments:
            boxes (Tensor[N, 6])
            scores (Tensor[N, 1])
        Returns:
            Fast NMS results
        '''
        #scores, idx = scores.sort(1, descending=True)
        #scores, idx = scores.sort(dim=0,descending=True)
        #print(type(scores))
        #print(scores)
        #scores = scores.unsqueeze(dim=1)
        #print(f'boxes shape:{boxes.shape}') [N, 6]
        #print(f'scores shape:{scores.shape}') [N, 1]
        
        scores, idx = scores.sort(dim=0,descending=True)
        # print(f'idx:{idx.shape}') #[N, 1] 
        # idx 為排序完的順序為原本的哪個位置
        idx = idx.squeeze(dim=1)
        # print(f'scores shape:{scores.shape}')
        boxes = boxes[idx]   # 对框按得分降序排列
        # print(f'boxes shape:{boxes.shape}')
        scores = scores
        # diou
        if cfg.TRAIN.NMS_IOU == 'diou':
            iou = self.calc_diou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
        # iou
        elif cfg.TRAIN.NMS_IOU == 'iou':
            iou = self.calc_iou(boxes, boxes).triu_(diagonal=1)  # IoU矩阵，上三角化
        C = iou
        for i in range(200):    
            A=C
            maxA = A.max(dim=0)[0]   # 列最大值向量
            E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 对角矩阵E的替代
            C = iou.mul(E)     # 按元素相乘
            if A.equal(C)==True:     # 终止条件
                break
        keep = maxA < NMS_threshold  # 列最大值向量，二值化
        # print(f'keep:{keep}')
        
        n = len(scores)
        weights = (C*(C>NMS_threshold).float() + torch.eye(n).cuda()) * (scores.reshape((1,n)))
        xx1 = boxes[:,0].expand(n,n)
        yy1 = boxes[:,1].expand(n,n)
        xx2 = boxes[:,2].expand(n,n)
        yy2 = boxes[:,3].expand(n,n)
        zz1 = boxes[:,4].expand(n,n)
        zz2 = boxes[:,5].expand(n,n)


        weightsum=weights.sum(dim=1)         # 坐标加权平均
        xx1 = (xx1*weights).sum(dim=1)/(weightsum)
        yy1 = (yy1*weights).sum(dim=1)/(weightsum)
        xx2 = (xx2*weights).sum(dim=1)/(weightsum)
        yy2 = (yy2*weights).sum(dim=1)/(weightsum)
        zz1 = (zz1*weights).sum(dim=1)/(weightsum)
        zz2 = (zz2*weights).sum(dim=1)/(weightsum)

        boxes = torch.stack([xx1, yy1, xx2, yy2, zz1, zz2], 1)
        #print(boxes)
        # torch.IntTensor(keep)
        # 原本為 keep
        return boxes[keep], keep, scores[keep]

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W, D) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        # input 的內容
        #rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
        #                         im_info, cfg_key))
        # print(f"score shape:{input[0].shape}")
        # (1, 9, 8, 8, 8)
        # 前9位是背景的概率，後9位是前景的概率
        scores = input[0][:, self._num_anchors:, :, :, :] # rpn_cls_prob.data 前景分數
        
        # (1, 54, 8, 8, 8) 9*6 anchor*(x1,y1,x2,y2,z1,z2)
        bbox_deltas = input[1] # rpn_bbox_pred.data
        # print(f"bbox_deltas:{bbox_deltas.shape}")
        im_info = input[2]
        cfg_key = input[3]
        # print(f'im_info:{im_info}')
        #print(f'cfg_key:{cfg_key}')

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N # 12000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # 2000
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH # 0.7
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        min_size      = cfg[cfg_key].RPN_MIN_SIZE # 8

        batch_size = bbox_deltas.size(0)
        
        feat_height, feat_width, feat_depth = scores.size(2), scores.size(3), scores.size(4)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_z = np.arange(0, feat_depth) * self._feat_stride

        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)
        # 產生偏移量
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel(),
                                             shift_z.ravel(), shift_z.ravel()))
                                  .transpose())

        shifts = shifts.contiguous().type_as(scores).float()

        # add A anchors (1, A, 6) to
        # cell K shifts (K, 1, 6) to get
        # shift anchors (K, A, 6)
        # reshape to (batch_size, K*A, 6) shifted anchors

        A = self._num_anchors # 9
        K = shifts.size(0) # (x*y*z,6) 8^3 
        # print(f'cell shift true:{K}')
        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 6) + shifts.view(K, 1, 6)
        anchors = anchors.view(1, K * A, 6).expand(batch_size, K * A, 6)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 4, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 6)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        # (x1,y1,x2,y2,z1,z2)
        proposals = clip_boxes(proposals, im_info, batch_size)

        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)
        
        # _, order = torch.sort(scores_keep, 1, True)
        
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        #  (n, x1, y1, x2, y2, z1, z2)
        # new 為隨機定一 m x n 大小的tensor
        output = scores.new(batch_size, post_nms_topN, 7).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            # (pre_nms_topN,(x1,y1,x2,y2,z1,z2))
            proposals_single = proposals_single[order_single, :]
            #　(,1)
            scores_single = scores_single[order_single].view(-1,1)
            #print(f"score single shape:{scores_single.shape}") # [4608, 1]
            #print(f"proposal single shape:{proposals_single.shape}")# [4608, 6]
            #print(f"proposals_single:{proposals_single}")
            #print(f"scores_single:{scores_single}")

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            
            anchorBoxes_f ,keep_idx_i, scores_f = self.Weighted_cluster_nms(proposals_single, scores_single, nms_thresh)
            # print(f"anchorBoxes_f:{anchorBoxes_f}")

            #print("NMS is OK")
            # 好像沒有將新的box當output 但之後可以判斷一下
            # print(f'keep idx:{keep_idx_i.shape}')
            # print(f'anchor box:{anchorBoxes_f.shape}')
            # print(f'score:{scores_f.shape}')
            #keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            # anchorBoxes_f ,anchors_nms_idx, scores_f = Weighted_cluster_nms(anchorBoxes, scores, 1e-5)
            
            keep_idx_i = keep_idx_i.long().view(-1)
            
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            #print(f"rois not zero number:{num_proposal}")
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
            
            #output[i,:,:] = anchorBoxes_f
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        ds = boxes[:, :, 5] - boxes[:, :, 4] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)) 
                & (ds >= min_size.view(-1,1).expand_as(ds)))
        return keep

    
