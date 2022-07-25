from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# 將anchor 設小一點 嘗試看看
# np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
# 2**np.arange(1, 4)
def generate_anchors(base_size=4, ratios=[0.5, 1, 2],
                     scales=2**np.arange(1, 4)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scale 跟Backbone 產生的feature map相關
    scales wrt a reference (0, 0, 15, 15) window.
    retrun anchor (x1,y1,x2,y2,z1,z2)
    """
    # x0,y0,x1,y1,z0,z1
    base_anchor = np.array([1, 1, base_size, base_size, 1, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios, base_size)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    d = anchor[5] - anchor[4] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    z_ctr = anchor[4] + 0.5 * (d - 1)
    return w, h, x_ctr, y_ctr, d, z_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr, ds, z_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis] # [[23],[16],[11]]
    hs = hs[:, np.newaxis] # [[12],[16],[22]]
    ds = ds[:, np.newaxis] # [[8],[16],[32]]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1),
                         z_ctr - 0.5 * (ds - 1),
                         z_ctr + 0.5 * (ds - 1)))
    return anchors

def _ratio_enum(anchor, ratios, base_size):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr, d, z_ctr = _whctrs(anchor)
    size = w * h 
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    ds = base_size * np.tile(ratios,1) # Z軸只需要將base_size照比例去除就好 
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr, ds, z_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr, d, z_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    ds = d * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr, ds, z_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
