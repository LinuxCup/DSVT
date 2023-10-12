import torch
import torch.nn as nn
from torch.autograd import Function


class MyUnique(Function):

    @staticmethod
    def forward(ctx, input):
        unique_flatten_inds, inverse = torch.unique(input, return_inverse=True)
        # out = nn.GELU()(input)
        return inverse

    @staticmethod
    def symbolic(g, input):
        return g.op("MyUnique", input)

    @staticmethod
    def backward(ctx, g):
        return None

myunique_ = MyUnique.apply

