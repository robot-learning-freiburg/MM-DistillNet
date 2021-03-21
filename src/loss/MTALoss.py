from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTALoss(nn.Module):
    def __init__(self, T=9.0, p=2.0):
        super(MTALoss, self).__init__()
        self.p = float(p)
        self.T = float(T)

    def forward(self, g_s, g_t):
        #return sum([self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])
        if torch.is_tensor(g_t[0]):
            # Just one element directly to handle
            return torch.stack([self.mtaloss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)], dim=0)
        else:
            # multiple elemtents, so g_t is of the form:
            # [ teacher_features, teacher_features ] == [[list of teacher features], [list of teacher features]]
            loss_list = []
            for i in range(len(g_s)):
                loss_list.append(
                    self.mtaloss(
                        g_s[i],
                        [f_t[i] for f_t in g_t]
                    )
                )
            return torch.stack(
                loss_list,
                dim=0
            )

    def mtaloss(self, out_s, out_t):

        # Calculate attention of student
        out_s = self.at(out_s)

        # In case out_t is a list and not a tensor, it means that we have
        # to calculate the P(X,Y)
        if not torch.is_tensor(out_t):
            # we assume it is a list of teachers, so calculate the attention
            # of all the teacher and then integrate by chain probability
            if len(out_t) == 1:
                out_t = self.at(out_t[0])
            else:
                # Need to make sure we multiply the normalized attention
                # attention added on item i
                out_t_reduce = self.at(out_t[0])
                for i in range(1, len(out_t)):
                    out_t_reduce = torch.mul(out_t_reduce,
                                             # Attention added on the new item i+1
                                             self.at(out_t[i]))

                out_t = torch.nn.functional.normalize(out_t_reduce, dim=1, p=1)
        else:
            # Provided a tensor directly, so just get the attention
            out_t = self.at(out_t)

        loss = F.kl_div(
            F.softmax(
                out_s/self.T,
                dim=1
            ),
            F.softmax(
                out_t/self.T,
                dim=1
            ),
            reduction='batchmean',
        )

        return loss

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
