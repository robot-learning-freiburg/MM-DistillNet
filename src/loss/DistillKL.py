"""
Taken from https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/AT.py
@inproceedings{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
"""
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, g_s, g_t):
        return sum([self.distill(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def distill(self, y_s, y_t):
        #y_t = Variable(y_t, requires_grad=False)
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
