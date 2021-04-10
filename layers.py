# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.sparse as sp

import torch

from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models
from utility.preprocessing import *

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        # assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        # assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        # assert not torch.isnan(h_prime).any()


        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.tanh(self.out_att(x, adj))
        return x
        # return self.out_att(x, adj)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, norm='', bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias
#         self.linear = nn.Linear(in_features, out_features, bias)
#         self.norm = norm
#
#     def forward(self, input, adj=1.0):
#         input = to_dense(input)
#         support = self.linear(input)
#         if isinstance(adj, (float, int)):
#             output = support * adj
#         else:
#             adj = adj_norm(adj, True) if self.norm == 'symmetric' else adj_norm(adj,
#                                                                                 False) if self.norm == 'asymmetric' else adj
#             output = torch.matmul(adj, support)
#
#         return output
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, norm='', bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear1 = nn.Linear(in_features, in_features, bias)
        self.linear1_ = nn.Linear(in_features, in_features, bias)
        self.linear = nn.Linear(in_features, out_features, bias)
        # self.linear2 = nn.Linear(in_features, out_features, bias)
        self.norm = norm


    def forward(self, input, adj=1.0):
        input = to_dense(input)
        # support = input
        if isinstance(adj, (float, int)):
            support = self.linear1(input)
            support_ = self.linear1_(input)
            output1 = F.tanh(support * adj)
            output2 = F.sigmoid(support_ * adj)
            output = torch.mul(output1, output2)
            return self.linear(output)
            # return self.linear(input)*adj
        else:
            support = self.linear1(input)
            support_ = self.linear1_(input)
            adj = adj_norm(adj, True) if self.norm == 'symmetric' else adj_norm(adj,
                                                                                False) if self.norm == 'asymmetric' else adj
            output1 = torch.matmul(adj, support)
            output2 = torch.matmul(adj, support_)
            output1 = F.tanh(output1)
            output2 = F.sigmoid(output2)
            output = torch.mul(output1, output2)
            output = self.linear(output)
            return output
            # output = torch.matmul(adj, input)
            # return self.linear(output)

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={}, bias={}, norm={})'.format(
            self.in_features, self.out_features, self.bias, self.norm)


class ImageGraphConvolution(nn.Module):
    """
    GCN layer for image data
    """

    def __init__(self, enc, inchannel=3):
        super(ImageGraphConvolution, self).__init__()
        self.encoder = enc
        self.classifier = nn.Linear(1024, 14)

    def forward(self, input, adj=1.0):
        x = self.encoder(input).squeeze()
        x = x.view(-1, 1024)
        support = self.classifier(x)
        if isinstance(adj, (float, int)):
            output = support * adj
        else:
            output = torch.spmm(adj, support)
        return output

class Transformer(nn.Module):
    def __init__(self, poolsize = 7, convsize = 2048):
        super(Transformer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(poolsize)
        self.fc1 = nn.Conv2d(convsize, convsize, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.gap(input)
        x = self.fc1(x)
        x = self.sig(x)
        x = input.mul(x)

        return x


class MyAlexNet(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        if inchannel != 3:
            original_model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.features = original_model.features
        self.conv1 = nn.Sequential(Transformer(6,256),nn.Conv2d(256, 1024, 3,  stride=1, padding=1), nn.BatchNorm2d(1024),
                                  nn.ReLU(inplace=True),nn.MaxPool2d(2, padding=1),nn.MaxPool2d(gpsize))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 1024, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),nn.MaxPool2d(2, padding=1),nn.MaxPool2d(gpsize))

        # self.features.add_module('transit', nn.Sequential(Transformer(6,256), nn.Conv2d(256, 1024, 3, padding=1), nn.BatchNorm2d(1024),
        #                                                   nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        # self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        # self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2
        # # x = x.view(-1, 1024)
        # # x = self.classifier(x)
        # return x


class MyResNet50(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)
        if inchannel != 3:
            original_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.conv1 = nn.Sequential(Transformer(7,2048),nn.Conv2d(2048, 1024, 3,  stride=1, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),nn.MaxPool2d(2, padding=1),nn.MaxPool2d(gpsize))
        self.conv2 = nn.Sequential(nn.Conv2d(2048, 1024, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True),nn.MaxPool2d(2, padding=1),nn.MaxPool2d(gpsize))
        # self.features.add_module('transit', nn.Sequential(nn.Conv2d(2048, 1024, 3, stride=1, padding=1),
        #                                        nn.BatchNorm2d(1024),nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        #
        # self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        # self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2
        # return x



class MyVggNet16_bn(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyVggNet16_bn, self).__init__()
        original_model = models.vgg16_bn(pretrained=True)
        if inchannel != 3:
            original_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.features = original_model.features
        self.conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1), nn.MaxPool2d(gpsize))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1), nn.MaxPool2d(gpsize))
        # self.features.add_module('transit', nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024),
        #                                                   nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        # self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        # self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        # x = self.features(x)
        # x = x.view(-1, 1024)
        # x = self.classifier(x)
        # return x
        x = self.features(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2


class MyVggNet16(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyVggNet16, self).__init__()
        original_model = models.vgg16(pretrained=True)
        if inchannel != 3:
            original_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.features = original_model.features
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                          nn.ReLU(inplace=True),
                                                          nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.features.add_module('transformer', Transformer())
        self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class MyDensNet161(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet161, self).__init__()
        original_model = models.densenet161(pretrained=True)
        if inchannel != 3:
            original_model.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)

        self.features = original_model.features
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(2208, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                          nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class MyDensNet201(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet201, self).__init__()
        original_model = models.densenet201(pretrained=True)
        if inchannel != 3:
            original_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)

        self.features = original_model.features
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(1920, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                          nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class MyDensNet121(nn.Module):
    def __init__(self, outnum=14, gpsize=4, inchannel=3):
        super(MyDensNet121, self).__init__()
        original_model = models.densenet121(pretrained=True)
        if inchannel != 3:
            original_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)

        self.features = original_model.features
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024),
                                                          nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(gpsize))
        self.classifier = nn.Linear(1024, outnum)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class DictReLU(nn.ReLU):
    def forward(self, input):
        return {key: F.relu(fea) for key, fea in input.items()} if isinstance(input, dict) else F.relu(input)


class DictDropout(nn.Dropout):
    def forward(self, input):
        if isinstance(input, dict):
            return {key: F.dropout(fea, self.p, self.training, self.inplace) for key, fea in input.items()}
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class DEDICOMDecoder(nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x: x):
        super(DEDICOMDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight_global = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.weight_local = Parameter(torch.FloatTensor(num_types, input_dim))
        self.reset_parameters()
        if issymmetric:
            self.weight_global = self.weight_global + self.weight_global.t()

    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight_global.size(1))
        self.weight_global.data.uniform_(-stdv, stdv)
        self.weight_local.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, type_index):
        relation = torch.diag(self.weight_local[type_index])
        product1 = torch.mm(input1, relation)
        product2 = torch.mm(product1, self.weight_global)
        product3 = torch.mm(product2, relation)
        outputs = torch.mm(product3, input2.transpose(0, 1))
        outputs = outputs + self.bias

        return self.act(outputs)


class DistMultDecoder(nn.Module):
    """DistMult Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_types, bias=True, act=lambda x: x):
        super(DistMultDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight = Parameter(torch.FloatTensor(num_types, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, type_index):
        relation = torch.diag(self.weight[type_index])
        intermediate_product = torch.mm(input1, relation)
        outputs = torch.mm(intermediate_product, input2.transpose(0, 1))
        outputs = outputs + self.bias

        return self.act(outputs)


class BilinearDecoder(nn.Module):
    """Bilinear Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x: x):
        super(BilinearDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.bias = Parameter(torch.rand(1)) if bias else 0
        self.weight = Parameter(torch.FloatTensor(num_types, input_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6. / self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, type_index):
        self.wt = self.weight + self.weight.transpose(1, 2) if issymmetric else self.weight
        intermediate_product = torch.mm(input1, self.wt[type_index])
        outputs = torch.mm(intermediate_product, input2.transpose(0, 1))
        outputs = outputs + self.bias
        return self.act(outputs)


class LinearDecoder(nn.Module):
    """Linear Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_types, issymmetric=True, bias=True, act=lambda x: x):
        super(LinearDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.layer = nn.Linear(input_dim, 1, bias) if issymmetric else nn.Linear(input_dim * 2, 1, bias)

    def forward(self, input1, input2, type_index):
        outputs = []
        for input in input2:
            if self.issymmetric:
                output = self.layer(input1) + self.layer(input.expand_as(input1))
            else:
                output = self.layer(torch.cat([input1, input.expand_as(input1)], dim=1))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return self.act(outputs)


class MLPDecoder(nn.Module):
    """multi-layer perceptron Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_types, hid_dim=20, issymmetric=True, bias=True, act=lambda x: x):
        super(MLPDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.issymmetric = issymmetric
        self.layer1 = nn.Linear(input_dim, hid_dim, bias) if issymmetric else nn.Linear(input_dim * 2, hid_dim, bias)
        self.layer2 = nn.Linear(hid_dim, 1, bias)

    def forward(self, input1, input2, type_index):
        outputs = []
        for input in input2:
            if self.issymmetric:
                output = self.layer1(input1) + self.layer1(input.expand_as(input1))
            else:
                output = self.layer1(torch.cat([input1, input.expand_as(input1)], dim=1))
            output = F.relu(output)
            outputs.append(self.layer2(output))
        outputs = torch.cat(outputs, dim=1)
        return self.act(outputs)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim=None, num_types=None, bias=True, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.num_types = num_types
        self.bias = Parameter(torch.rand(1)) if bias else 0

    def forward(self, input1, input2, type_index=None):
        outputs = torch.mm(input1, input2.transpose(0, 1))
        outputs = outputs + self.bias
        return self.act(outputs)