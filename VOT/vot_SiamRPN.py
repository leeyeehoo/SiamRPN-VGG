# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# load net
net_file = '/home/lee/tracking/challenge-test/vot-toolkit/tracker/examples/SiamVGGRPN/code/SiamRPN0328checkpoint.pth.tar'

net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file)['state_dict'])
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_file)  # HxWxC
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    state = SiamRPN_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))

