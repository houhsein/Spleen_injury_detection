import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config_3d import cfg
from model.rpn.rpn_3d import _RPN
from model.roi_align import RoIAlign3D
#from model.roi_layers import ROIAlign, ROIPool
#from model.roi_pooling.modules.roi_pool import _RoIPooling
#from model.roi_crop.modules.roi_crop import _RoICrop
#from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade_3d import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils_3d import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, FocalLoss, compute_diou

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_align = RoIAlign3D(cfg.POOLING_SIZE, cfg.POOLING_SIZE_Z, 1.0/16.0, 1.0/16.0) 
        
        #self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0) # POOLING_SIZE 7
        #self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        #self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE # POOLING_SIZE 7
        #self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        # print(f'batch size:{batch_size}')
        # print(f'im_data shape:{im_data.shape}')
        # im_info = im_info.data
        # gt_boxes = gt_boxes.data
        # num_boxes = num_boxes.data always 2 spleen & background

        im_info = im_info # (2,1,128,128,128) 可能要改batch size
        gt_boxes = gt_boxes
        num_boxes = num_boxes

        # feed image data to base model to obtain base feature map
        # Resnet model
        base_feat = self.RCNN_base(im_data)
        #print(f'base feat:{base_feat.shape}') [b,1024,8,8,8]
        # check base_feat is all greater than 0
        # idex = torch.gt(base_feat,0)
        # print(base_feat[idex].view(base_feat.shape))
        
        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        #print(f"RPN rois shape:{rois.shape}")
        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        #print(f"rois shape:{rois.shape}") # [B, 128, 7] 正常的
        #print(f"rois:{rois}")
        #print(f"rois_target:{rois_target.shape}") # [512, 6]
        #print(f"rois_target:{rois_target}")
        #print(f"rois_label:{rois_label}")
        #print(f"rois inside ws bigger than 0:{rois_inside_ws.ge(0).any()}")
        #print(f"rois outside ws bigger than 0:{rois_outside_ws.ge(0).any()}")
        # do roi pooling based on predicted rois 
        # (only roi_align can be used)
        '''
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 7))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,7))
        '''

        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 7))
        # print(f'ROI align output shape:{pooled_feat.shape}')
        # feed pooled features to top model
        # fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        pooled_feat = self._head_to_tail(pooled_feat)
        #print(f'pooled_feat shape:{pooled_feat.shape}') # [512,2048]

        # Classification

        # compute bbox offset 
        # nn.Linear(2048, 6 * self.n_classes)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat) # [512,12]
        #print(f'bbox_pred:{bbox_pred}')
        #print(f'bbox_pred shape:{bbox_pred.shape}') # [512,12]
        #print(f'rois_label:{rois_label}')
        #print(f'rois_label shpae:{rois_label.shape}')
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 6), 6)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 6))
            bbox_pred = bbox_pred_select.squeeze(1)
        #print(f"Selected bbox_pred:{bbox_pred.shape}")
        #print(f"Selected bbox_pred:{bbox_pred}")
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            # Cross entropy
            if cfg.TRAIN.CLASS_LOSS == 'cross_entropy':
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # Focal loss
            elif cfg.TRAIN.CLASS_LOSS == 'focal_loss':
                focal_loss = FocalLoss(class_num=2, alpha=0.25, gamma=2)
                RCNN_loss_cls = focal_loss(cls_score, rois_label)
            else:
                print('Classification loss only cross_entropy or focal_loss')
            # bounding box regression L1 loss
            #print(f"----------- RCNN box regression -----------")
            # regression loss
            # L1 loss function 
            if cfg.TRAIN.BOX_REGRESSION_LOSS == 'L1':
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            elif cfg.TRAIN.BOX_REGRESSION_LOSS == 'diou':
                _, RCNN_loss_bbox = compute_diou(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            else:
                print('Regression loss only L1 or diou')

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
    
    # init_weight 不確定怎麼init 
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
