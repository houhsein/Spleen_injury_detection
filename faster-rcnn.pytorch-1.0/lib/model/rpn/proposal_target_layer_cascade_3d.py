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
import numpy.random as npr
from ..utils.config_3d import cfg
from .bbox_transform_3d import bbox_overlaps_batch, bbox_transform_batch
import pdb


# 该函数将rois和gt boxes结合起来，对产生的rois进行筛选和分类（每一个roi中的目标属于哪一种类别）。
# 同时产生bbox inside weights和bbox outside weights，用以loss值的确定。
class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS) # (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) # (0.1, 0.1, 0.2, 0.2, 0.1, 0.2)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS) # (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        
    def forward(self, all_rois, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        # self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        # gt_boxes 是int 所以會有問題!!!!
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.cuda()
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        #print(f"BBOX_NORMALIZE_MEANS:{self.BBOX_NORMALIZE_MEANS}")
        #print(f"BBOX_NORMALIZE_STDS:{self.BBOX_NORMALIZE_STDS}")
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        # 輸出的gt boxes: [label,x1,y1,x2,y2,z1,z2]
        gt_boxes_append[:,:,1:7] = gt_boxes[:,:,:6]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        # Minibatch size (number of regions of interest [ROIs])
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) #  BATCH_SIZE 128
        #FG_FRACTION为前景,背景比例
        # 平均每张图片上的前景rois数目
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)) # FG_FRACTION 0.25
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        #print(f"fg_rois_per_image:{fg_rois_per_image}")
        #print(f"all_rois:{all_rois.shape}")
        # Sample rois with classification labels and bounding box regression targets

        # 对所有的rois进行采样，选区其中的一部分作为前景rois，背景rois，
        # 返回他们的labels标签，rois，bbox回归的目标矩阵和bbox inside weights
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, # 所有的rois，包括产生的proposals和gt boxes
            gt_boxes, 
            fg_rois_per_image, # 每张图片的前景rois数目 32
            rois_per_image, # 平均每张图片上的rois总数目 128
            self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th, tz, td)

        This function expands those targets into the 6-of-6*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 6K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 6K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 6).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 6
        assert gt_rois.size(2) == 6

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)
        #print(f"compute target:{targets}")
        #print(f'compute target:{targets.shape}') # [4, 128, 6]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)) # (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets)) # (0.1, 0.1, 0.1, 0.2, 0.2, 0.2)

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
 
        # 计算所有的产生的rois和gt boxes之间的overlaps（IOU）。
        # overlaps是一个shape为[N, K]的二维数组，K表示所有的rois的数目，N表示gt boxes的数目。
        # 对应overlap[i, j]存放的是第i个gt boxes和第j个roi之间的IOU。
        #print(f'all_rois shape:{all_rois.shape}')
        #print(f'all_rois:{all_rois}')
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes) # [gt_num, rois_num, 1] 
        # print(f"RCNN overlaps shape:{overlaps.shape}")
        # 取ground ture中最大的IOU(多個ground ture)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2) # alway is 1 我們的資料一張圖只有一個gt box
        # print(f"RCNN max_overlaps num:{num_proposal}")
        #print(f"RCNN overlaps is bigger than 0.5:{max_overlaps.ge(0.5).any()}")
        # print(f"RCNN max_overlaps shape:{max_overlaps.shape}")
        
        # print(f"RCNN num_boxes_per_img :{num_boxes_per_img}")

        offset = torch.arange(0, batch_size)* gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        #print(f"offset shape:{offset.shape}")
        # labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        labels = gt_boxes[:,:,6].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        #print(f"RCNN labels shape:{labels.shape}")
        #print(f"RCNN labels is all 1:{labels.eq(1).all()}")
        labels_batch = labels.new(batch_size, rois_per_image).zero_() #[4,128]
        rois_batch  = all_rois.new(batch_size, rois_per_image, 7).zero_() # [4,128,7]
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 7).zero_() # [4,128,7]

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1) # FG_THRESH 0.5
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) & # BG_THRESH_HI 0.5
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1) # BG_THRESH_LO 0.1
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
            print(f"RCNN rios num:fg/bg=({fg_num_rois}/{bg_num_rois})")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # 學到gt box
            #print(f"RCNN keep_inds:{keep_inds.shape}")
            #print(f"RCNN keep_inds:{keep_inds}")
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            #print(f"RCNN labels_batch:{labels_batch}")

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i
            #print(f"rois_batch:{rois_batch}")
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:7], gt_rois_batch[:,:,:6])
        #print(f"bbox_target_data:{bbox_target_data}")
        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
