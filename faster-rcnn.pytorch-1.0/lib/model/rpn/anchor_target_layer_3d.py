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

from model.utils.config_3d import cfg
from .generate_anchors_3d import generate_anchors
from .bbox_transform_3d import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
        :param rpn_cls_prob: tensor of shape(B, A*2, H, W, D)
        :param gt_boxes: tensor of shape(B, N, 7) [x1,y1,x2,y2,z1,z2, class]
        :param im_info: tensor of shape(B, 3) [x,y,z]
        :return:
        rpn_label_target: tensor of shape(B, A*2, H, W, D) not correct
        rpn_bbox_target: tensor of shape(B, A*6, H, W, D)
        rpn_bbox_inside_weights: tensor of shape(B, A*6, H, W, D)
        rpn_bbox_outside_weights: tensor of shape(B, A*6, H, W, D)
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        self._ratios = ratios
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0
    
    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
        # input information
        # [2, 18, 8, 8, 8]
        #print(f"anchor scales:{self._scales}")
        #print(f"anchor ratios:{self._ratios}")
        rpn_cls_score = input[0] # rpn_cls_score.data 
        gt_boxes = input[1] # gt_boxes (batch_size, 1, (x1,y1,x2,y2,z1,z2))
        #print(f'gt_boxes:{gt_boxes}')
        im_info = input[2] # [b,[x,y,z]]
        num_boxes = input[3]

        # 要確定in_info內容
        # [b,[x,y,z]]
        # print(f'im_info:{im_info}')
        # map of shape (..., H, W)
        height, width, depth = rpn_cls_score.size(2), rpn_cls_score.size(3), rpn_cls_score.size(4)
        '''
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width , 'depth', depth
            print ''
            print 'im_size: ({}, {}, )'.format(im_info[0][0], im_info[0][1])
            print 'scale: {}'.format(im_info[0][2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes
        '''
        batch_size = gt_boxes.size(0)
        # print(f"feat stride:{self._feat_stride}") # 4 
        feat_height, feat_width, feat_depth = rpn_cls_score.size(2), rpn_cls_score.size(3), rpn_cls_score.size(4)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_z = np.arange(0, feat_depth) * self._feat_stride

        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel(),
                                             shift_z.ravel(), shift_z.ravel() ))
                                             .transpose())
        # nn tensor 經形狀或轉換會變成非連續性，利用.contiguous() 轉換，以利之後view操作
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()
        # print(f"shfits:{shifts}")

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 6) + shifts.view(K, 1, 6)
        all_anchors = all_anchors.view(K * A, 6)
        # print(f'All anchors shape:{all_anchors.shape}') [4068,6]
        total_anchors = int(K * A)

        # only keep anchors inside the image
        # (all_anchors[:, 5] < long(im_info[0][0]) + self._allowed_border) 
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 4] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) & # width
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border) & # height
                (all_anchors[:, 5] < long(im_info[0][2]) + self._allowed_border))

        # 確定一下keep內容
        # print(f"keep shape:{keep.shape}")
        inds_inside = torch.nonzero(keep).view(-1)
        #print(f'select num inside:{len(inds_inside)}')
        '''
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)
        '''
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        #print(f"Inside anchor num: {anchors.size(0)}")
        '''
        if DEBUG:
            print 'anchors.shape', anchors.shape
        '''

        # 與 ground tures比較 開始label positive和negative
        # label: 1 is positive, 0 is negative, -1 is dont care
        # 剛開始將所有label設置為-1
        # 隨機新增tensor 不論inputs是多少维的，新建的new_inputs的type和device都与inputs保持一致
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_().float()
        # 要確定 overlaps 內容
        # (batch_size, N, K) ndarray of overlap between boxes and query_boxes
        # N個anchor去比對到K的ground truth的overlap area
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        # print(f'overlap shape:{overlaps.shape}') # (batch_size,anchor number,ground ture number)
        
        # 對於每個anchor 取與ground truth之間最大的IOU (多個ground ture)
        # max_overlaps 為value argmax_overlaps為index shape皆為(batch_size,N)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        #print(f'Max overlaps shape:{max_overlaps.shape}')
        # 對於每個ground truth，取與anchor之間最大的IOU 
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        #print(f'Anchor max IOU shape:{max_overlaps.shape}')
        #print(f'Anchor max IOU :{max_overlaps}')
        #print(f'GT max IOU :{max_overlaps}')
        # If an anchor statisfied by positive and negative conditions set to negative
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES: # False
        # < 0.01
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # 與anchor overlap比較，把所有anchor對應最大IOU的個數加總 
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        #print(f' Keep shape:{keep.shape}') # (B, N)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        # > 0.3
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        
        
        # Max number of foreground examples
        # Total number of examples
        # 0.5 * 256
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        # print(f'Final labels:{labels}')   

        # 因應多batch的算法
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        
        # bbox_targets本质上是RPN应该生成的数据，用以bbox的回归操作。即bbox targets是网络在训练的时候需要生成的目标。
        # gt_boxes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7) 
        # gt_box * N shape [B, N, 7]
        # tensor([[[ 38,  16,  94,  ...,  93, 127,   0],
        # [ 38,  16,  94,  ...,  93, 127,   0],
        # [ 38,  16,  94,  ...,  93, 127,   0],

        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7))

        #print(f'bbox_targets fg shape:{bbox_targets.shape}')
        #print(f'bbox_targets fg:{bbox_targets[labels == 1, :]}')

        # use a single value instead of 6 values for easy index.
        # only the positive ones have regression targets

        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0] # 1
        #print(f'bbox_inside_weights"{bbox_inside_weights[labels==1].shape}')
        #print(f"bbox_inside_weights:{bbox_inside_weights[labels==1]}")
        # bbox_outside_weights 相當於公式中的 lamda(1/N_{reg})，用來平衡分類和回歸loss
        # 保证cfg.TRAIN.RPN_POSITIVE_WEIGHT 在0和1之间
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0: # -1
            num_examples = torch.sum(labels[i] >= 0)
            # print(f"num of fg & bg:{num_examples}")
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        # 对正负样本（anchors）的bbox outside weights分别进行赋值。
        #print(f"bbox_inside_weights type:{bbox_inside_weights.dtype}")
        #print(f"bbox_outside_weights type:{bbox_outside_weights.dtype}")
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        #bbox_outside_weights[labels == 1] = 1/256
        #bbox_outside_weights[labels == 0] = 1/256
        #print(f"positive_weights:{positive_weights}")
        #print(f"bbox_outside_weight pos:{bbox_outside_weights[labels == 1]}")
        #print(f"bbox_outside_weight neg:{bbox_outside_weights[labels == 0]}")
        #print(f"Unmap bbox_outside_weights:{bbox_outside_weights.ne(0).any()}")
        '''
        if DEBUG:
            _sums = np.zeros((1, 4))
            _squared_sums = np.zeros((1, 4))
            _counts = 1e-14
            _sums += bbox_targets[labels == 1, :].sum(axis=0)
            _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            _counts += np.sum(labels == 1)
            means = _sums / _counts
            stds = np.sqrt(_squared_sums / _counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds
        '''

        # 将剩下的anchors(outside anchor)进行label的标注，由于这些anchors并不是全部存在于图片内部，因此这里将他们的label设置为-1，表示不关心。
        # total_anchors是一个整数，表示之前生成的最最原始的anchors的数目。
        # inds_inside表示存在于图片内部的anchors在所有原始anchors中的索引。
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill= -1)
        # 下面的三行代码也是同理，将那些原始的anchors的信息也添加进去，由于没有经过上面的计算过程，因此这些信息全部被设置为0。
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        #print(f'bbox_targets unmap fg :{bbox_targets[labels == 1, :].shape}')
        #print(f'bbox_targets unmap bg :{bbox_targets[labels == 0, :].shape}')
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # print(f"Maped bbox_outside_weights:{bbox_outside_weights.ne(0).any()}")
        '''
        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            _fg_sum += np.sum(labels == 1)
            _bg_sum += np.sum(labels == 0)
            _count += 1
            print 'rpn: num_positive avg', _fg_sum / _count
            print 'rpn: num_negative avg', _bg_sum / _count
        '''
        outputs = []

        # 将bbox_target，bbox_inside_weights和bbox_outside_weights resize为规定大小
        # 这里height，width分别表示特征图的高度和宽度，A表示在每个位置产生的anchors的数目，reshape之后就和特征图上的位置一一对应。
        #print(f"label:{labels.shape}")
        labels = labels.view(batch_size, height, width, depth, A).permute(0,4,1,2,3).contiguous()
        labels = labels.view(batch_size, 1, A * height * depth, width)
        #print(f"final label:{labels.shape}")
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, depth, A*6).permute(0,4,1,2,3).contiguous()
        #print(f"final bbox_target:{bbox_targets.shape}")
        outputs.append(bbox_targets)
        

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 6)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, depth, 6*A)\
                              .permute(0,4,1,2,3).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 6)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, depth, 6*A)\
                              .permute(0,4,1,2,3).contiguous()
        #print(f"bbox_outside_weights resize:{bbox_outside_weights.ne(0).any()}")
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    
    # 維度为2，表示这里补完的是labels信息
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        # 这里补完的是bbox回归目标，权值等信息，和填充labels的过程类似。
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :6])
