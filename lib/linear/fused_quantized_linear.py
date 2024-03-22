import torch
import torch.nn as nn
import quiptools_cuda
from quip.lib.utils import dtype_from_str, get_hadK
from quip.lib import codebook
from quip.lib.linear.quantized_linear import QuipQuantLinear
import time


class FusedQuipQuantLinear(QuipQuantLinear):

    def __init__(self, fuse_dim, fuse_sizes, *QL_args, **QL_kwargs):
        super(FusedQuipQuantLinear, self).__init__(*QL_args, **QL_kwargs)
        self.fuse_dim = fuse_dim
        self.fuse_sizes = fuse_sizes
        self.register_buffer('fuse_scales', torch.ones(len(self.fuse_sizes)))
        self.n = len(self.fuse_sizes)

    def forward(self, input):
        fused_output = super(FusedQuipQuantLinear, self).forward(input)
        split_outputs = torch.split(fused_output, self.fuse_sizes, self.fuse_dim)
        return tuple(split_outputs[i] * self.fuse_scales[i] for i in range(self.n))
