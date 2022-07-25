from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_3d import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    extend = 1

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(planes)
        )

        self.shrotcut = shortcut
        self.Relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, input):
        tmp = input
        out = self.layer(input)

        if self.shrotcut is not None:
            tmp = self.shrotcut(input)
        # print("out:",out.shape)
        # print("tmp:",tmp.shape)
        out = out + tmp
        out = self.Relu(out)

        return out

class Bottleneck(nn.Module):
    extend = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        """
        self.layer = nn.Sequential(
            self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=1,stride=stride, padding=0, bias=False),
            self.bn1 = nn.BatchNorm3d(out_channel),
            self.relu1 = nn.ReLU(inplace=True),
            self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            self.bn2 = nn.BatchNorm3d(out_channel),
            self.relu2 = nn.ReLU(inplace=True),
            self.conv3 = nn.Conv3d(out_channel, out_channel * 4, kernel_size=1, stride=stride, padding=0, bias=False),
            self.bn3 = nn.BatchNorm3d(out_channel * 4),
            #nn.ReLU(True)
        )
        """
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
    
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(True)
        self.shortcut = shortcut
        self.stride = stride
        

    def forward(self, input):
        tmp = input
        #out = self.layer(input)
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            tmp = self.shortcut(input)

        #　print("out:",out.shape)
        #　print("tmp:",tmp.shape)
        out = out + tmp
        out = self.relu(out)

        return out

'''
class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 1), padding=3, bias=False)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64*8, layers[3], stride=2)
        
        # C2, C3, C4, C5
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels, 
                         self.layer3[layers[2] - 1].conv2.out_channels, self.layer4[layers[3] - 1].conv2.out_channels]
            #fpn_sizes = [128 * block.extend, 256 * block.extend, 512 * block.extend]   
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, 
                         self.layer3[layers[2] - 1].conv3.out_channels, self.layer4[layers[3] - 1].conv3.out_channels]
            #fpn_sizes = [64* block.extend, 128 * block.extend, 256 * block.extend, 512 * block.extend]              
        else:
            raise ValueError(f"Block type {block} not understood")
'''
# only detect spleen,so class is one 
class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=2):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
    self.bn1 = nn.BatchNorm3d(64)
    self.relu = nn.ReLU(inplace=True)
    #self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)
    self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 64*2, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 64*4, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 64*8, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool3d(7)
    self.fc = nn.Linear(512 * block.extend, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        # "kaiming_uniform" type
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
          fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
          bound = 1 / np.sqrt(fan_out)
          nn.init.normal_(m.bias, -bound, bound)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    shortcut = None
    if stride != 1 or self.inplanes != planes * block.extend:
        shortcut = nn.Sequential(
            nn.Conv3d(self.inplanes, planes * block.extend,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes * block.extend),
        )
    # 重複固定的次數
    layers = []
    layers.append(block(self.inplanes, planes, stride, shortcut))
    self.inplanes = planes * block.extend
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

  def forward(self, inputs):
    if self.training:
        img_batch, annotations = inputs
    else:
        img_batch = inputs
    #print("img_batch:",img_batch.shape)
    x = self.conv1(img_batch)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    
    x5 = self.avgpool(x4)
    x5 = x.view(x5.size(0), -1)
    x5 = self.fc(x5)
    return x5
'''
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x
'''

def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    print('pre-train')
    model.load_state_dict('/tf/jacky831006/faster-rcnn.pytorch-0.4/pre-trian-weight/r3d101_K_200ep.pth')
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model


def resnet_test1(pretrained=False):
  """Constructs a ResNet_test1 model.
    block reduce last block, only 3 block but should change other module
    block: [3, 8 ,39] 
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

# class-agnostic 只回归2类bounding box 即前景和背景再结合每个box在classification 网络中对应着所有类别的得分
# class-specific 利用每一个RoI特征回归出所有类别的bbox坐标，並根據classification 结果索引到对应类别的box输出
class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    # pre-train model just use resnet 
    self.model_path = '/tf/jacky831006/faster-rcnn.pytorch-0.4/pre-trian-weight/r3d101_K_200ep.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.num_layers = num_layers

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.num_layers == 18:
        resnet = resnet18()
    if self.num_layers == 34:
        resnet = resnet34()     
    if self.num_layers == 50:
        resnet = resnet50()
    if self.num_layers == 152:
        resnet = resnet152()
    if self.num_layers == "test1":
        resnet = resnet_test1()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict(state_dict,strict=False)
      #resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()},strict=False)

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
      resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)
    if self.num_layers == "test1" :
      self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool, resnet.layer1, resnet.layer2)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 6)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 6 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    # Number of fixed blocks during training, by default the first of all 4 blocks is fixed
    # Range: 0 (none) to 3 (all)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False
    # apply function to be applied to each submodule
    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    #fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    # 對 x,y,z 取 mean [256,2048,4,4,4] -> [256,2048]
    fc7 = self.RCNN_top(pool5).mean(4).mean(3).mean(2)
    return fc7
