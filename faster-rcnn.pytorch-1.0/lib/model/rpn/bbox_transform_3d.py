# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    '''
    计算两个N * 6的矩阵之间的相关回归矩阵。
    本质上是在求解每一个anchor相对于它的对应gt box的（dx, dy, dw, dh, dz, dd）的六个回归值，返回结果的shape为[N, 6]。
    :param ex_rois: shape为[N, 6]的数组，一般传入的anchors的信息。
    :param gt_rois: shape为[N, 6]的数组，一般传入的gt boxes(ground truth boxes)的信息。每一个gt roi都与一个ex roi相对应。
    :return: 本质上是在求解每一个anchor相对于它的对应gt box的（dx, dy, dw, dh, dz, dd）的六个回归值，返回结果的shape为[N, 6]。
    '''

    # 求出每一个ex_roi的宽度高度和中心坐标
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_depths = ex_rois[:, 5] - ex_rois[:, 4] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    ex_ctr_z = ex_rois[:, 4] + 0.5 * ex_depths

    # 求解每一个gt box的宽度高度和中心坐标
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_depths = gt_rois[:, 5] - gt_rois[:, 4] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_ctr_z = gt_rois[:, 4] + 0.5 * gt_depths

    # 反向求解RPN网络应该生成的数据，即bbox进行回归操作需要的6个变量。
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = (gt_ctr_z - ex_ctr_z) / ex_depths
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    targets_dd = torch.log(gt_depths / ex_depths)

    # 将所有的信息组合成一个矩阵返回
    # vstack将所有的向量组合成shape为[6, N]的矩阵，最后transpose，矩阵的shape变成[N, 6]。
    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_dz, targets_dd),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):
    # ex_rois: [N,6]
    # gt_rois: [B,N,7] -> [B,N,(x1,y1,x2,y2,z1,z2,label)]
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_depths = ex_rois[:, 5] - ex_rois[:, 4] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_ctr_z = ex_rois[:, 4] + 0.5 * ex_depths

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_depths = gt_rois[:, :, 5] - gt_rois[:, :, 4] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_ctr_z = gt_rois[:, :, 4] + 0.5 * gt_depths

        # ex_ctr_x.view(1,-1).expand_as(gt_ctr_x): (H*W,)_>(1,H*W)_>(B,H*W) / (H*W,)
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dz = (gt_ctr_z - ex_ctr_z.view(1,-1).expand_as(gt_ctr_z)) / ex_depths
        
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))
        targets_dd = torch.log(gt_depths / ex_depths.view(1,-1).expand_as(gt_depths))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_depths = ex_rois[:,:, 5] - ex_rois[:,:, 4] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights
        ex_ctr_z = ex_rois[:, :, 4] + 0.5 * ex_depths


        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_depths = gt_rois[:, :, 5] - gt_rois[:, :, 4] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_ctr_z = gt_rois[:, :, 4] + 0.5 * gt_depths


        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dz = (gt_ctr_z - ex_ctr_z) / ex_depths
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
        targets_dd = torch.log(gt_depths / ex_depths)

    else:
        raise ValueError('ex_roi input dimension is not correct.')
    #targets (B,H*W,6)
    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_dz, targets_dd),2)

    return targets

def bbox_transform_inv(boxes, deltas, batch_size):
    # 這邊都要測試一下
    '''
    将boxes使用rpn网络产生的deltas进行变换处理，求出变换后的boxes，即预测的proposals。
    此处boxes一般表示原始anchors，即未经任何处理仅仅是经过平移之后产生测anchors。
    :param boxes: 一般表示原始anchors，即未经任何处理仅仅是经过平移之后产生测anchors，shape为[N, 4]，N表示anchors的数目。
    :param deltas: RPN网络产生的数据，shape为[N, (1 + classes) * 4]，classes表示类别数目，1 表示背景，N表示anchors的数目。
    :return: 预测的变换之后的proposals（或者叫anchors）
    '''

    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    depths = boxes[:, :, 5] - boxes[:, :, 4] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    ctr_z = boxes[:, :, 4] + 0.5 * depths

    # 获取每一个类别的deltas的信息，每一个类别的deltas的信息是顺序存储的，
    # 即第一个类别的六个信息（dx, dy, dw, dh, dz, dd）存储完成后才接着另一个类别。
    # 下面六个变量的shape均为[N, classes + 1]，N表示anchors数目，classes表示类别数目（此处为20），1表示背景。

    dx = deltas[:, :, 0::6]
    dy = deltas[:, :, 1::6]
    dz = deltas[:, :, 4::6]
    dw = deltas[:, :, 2::6]
    dh = deltas[:, :, 3::6]
    dd = deltas[:, :, 5::6]
    
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_ctr_z = dz * depths.unsqueeze(2) + ctr_z.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_d = torch.exp(dd) * depths.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::6] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::6] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::6] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::6] = pred_ctr_y + 0.5 * pred_h
    # z1
    pred_boxes[:, :, 4::6] = pred_ctr_z - 0.5 * pred_d
    # z2
    pred_boxes[:, :, 5::6] = pred_ctr_z + 0.5 * pred_d

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    将proposals的边界限制在图片内
    boxes: 等待裁剪的boxes，一般是一系列的anchors。
    return: 裁减之后的boxes，每一条边都在图片大小的范围内
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1
    batch_z = im_shape[:, 2] - 1


    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y
    boxes[:,:,4][boxes[:,:,4] > batch_z] = batch_z
    boxes[:,:,5][boxes[:,:,5] > batch_z] = batch_z

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::6].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::6].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::6].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::6].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,4::6].clamp_(0, im_shape[i, 2]-1)
        boxes[i,:,5::6].clamp_(0, im_shape[i, 2]-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 6) ndarray of float
    gt_boxes: (K, 6) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1) *
                (gt_boxes[:,5] - gt_boxes[:,4] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)
                (anchors[:,5] - anchors[:,4] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 6).expand(N, K, 6)
    query_boxes = gt_boxes.view(1, K, 6).expand(N, K, 6)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    idd = (torch.min(boxes[:,:,5], query_boxes[:,:,5]) -
        torch.max(boxes[:,:,4], query_boxes[:,:,4]) + 1)
    idd[idd < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih * idd)
    overlaps = iw * ih * idd / ua

    return overlaps


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 6) ndarray of float
    gt_boxes: (b, K, 7) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    (batch_size, N, K) ndarray of overlap between boxes and query_boxes
     N個anchor去比對到K的ground ture的overlap area
     我們的資料K都為1
    """
    batch_size = gt_boxes.size(0)

    # 判斷dim anchor有沒有 batch size
    # anchor 都沒有設 batch size
    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 6).expand(batch_size, N, 6).contiguous()
        gt_boxes = gt_boxes[:,:,:6].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_z = (gt_boxes[:,:,5] - gt_boxes[:,:,4] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y * gt_boxes_z).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_boxes_z = (anchors[:,:,5] - anchors[:,:,4] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y * anchors_boxes_z).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) & (gt_boxes_z == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1) & (anchors_boxes_z == 1)

        boxes = anchors.view(batch_size, N, 1, 6).expand(batch_size, N, K, 6).float()
        query_boxes = gt_boxes.view(batch_size, 1, K, 6).expand(batch_size, N, K, 6).float()

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0

        idd = (torch.min(boxes[:,:,:,5], query_boxes[:,:,:,5]) -
            torch.max(boxes[:,:,:,4], query_boxes[:,:,:,4]) + 1)
        idd[idd < 0] = 0

        ua = anchors_area + gt_boxes_area - (iw * ih * idd)
        overlaps = iw * ih * idd / ua

        # mask the overlap here.
        # 用value填充tensor中与mask中值为1位置相对应的元素 
        # 將ground ture 為 1 的當作0, anchor box為 1 的當作-1
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        # anchor 有batch size: (b,N,6)
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 6:
            # 不要取到class
            anchors = anchors[:,:,:6].contiguous()
        else:
            anchors = anchors[:,:,1:7].contiguous()

        gt_boxes = gt_boxes[:,:,:6].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_z = (gt_boxes[:,:,5] - gt_boxes[:,:,4] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y * gt_boxes_z).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_boxes_z = (anchors[:,:,5] - anchors[:,:,4] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y * anchors_boxes_z).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) & (gt_boxes_z == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1) & (anchors_boxes_z == 1)

        boxes = anchors.view(batch_size, N, 1, 6).expand(batch_size, N, K, 6).float()
        query_boxes = gt_boxes.view(batch_size, 1, K, 6).expand(batch_size, N, K, 6).float()

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0

        idd = (torch.min(boxes[:,:,:,5], query_boxes[:,:,:,5]) -
            torch.max(boxes[:,:,:,4], query_boxes[:,:,:,4]) + 1)
        idd[idd < 0] = 0

        ua = anchors_area + gt_boxes_area - (iw * ih * idd)
        
        # Intersection (iw * ih) divided by Union (ua)
        overlaps = iw * ih * idd / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
