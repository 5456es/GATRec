import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn


class RGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names,args):
        super().__init__()

     
        self.layers = 0
  
        self.heads=args.heads
        self.in_feats = in_feat
        self.hid_feats = hidden_feat
        self.out_feats = out_feat

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.in_feats, self.hid_feats // self.heads[0], self.heads[0])
            for rel in rel_names}, aggregate='mean')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // self.heads[1], self.heads[1])
            for rel in rel_names}, aggregate='mean')
        
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // self.heads[2], self.heads[2])
            for rel in rel_names}, aggregate='mean')

        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.out_feats // self.heads[3], self.heads[3])
            for rel in rel_names}, aggregate='mean')

        

        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for i in range(3)])
        self.bns2 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for i in range(3)])

    def residual(self,h,block_before,block_after):
        for type in block_after.srcdata['_ID']:
            for i in block_after.srcdata['_ID'][type]:
                index = block_before.srcdata['_ID'][type].index(i)
                
                h[type][i] = h[type][i] + block_before.srcdata['features'][type][index]

    def residual_2(self, h, block_before, block_after):
        for ntype in block_after.srcdata['_ID']:
            before_ids = block_before.srcdata['_ID'][ntype] ###[0,1,8,23,34,43，53]
            after_ids = block_after.srcdata['_ID'][ntype] ### [1,34,53]

            # 使用字典来查找索引
            id_to_index = {int(id_): idx for idx, id_ in enumerate(before_ids)} ### {0: 0, 1: 1, 8: 2, 23: 3, 34: 4, 43: 5, 53: 6}
            indices = torch.tensor([id_to_index[int(id_)] for id_ in after_ids], dtype=torch.long) 
            ### tensor([1, 4, 6])

            # 获取需要相加的特征
            before_features = block_before.srcdata['features'][ntype][indices]

            # 批量相加
            h[ntype] += before_features
        return h


    def forward(self, blocks, inputs):
        
        h = self.conv1(blocks[0], inputs)
        self.rel_list = list(h.keys())

        
        h[self.rel_list[0]] = F.leaky_relu(self.bns[0](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.leaky_relu(self.bns2[0](h[self.rel_list[1]].view(-1, self.hid_feats)))
        

        h = self.conv2(blocks[1], h)

        h[self.rel_list[0]] = F.leaky_relu(self.bns[1](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.leaky_relu(self.bns2[1](h[self.rel_list[1]].view(-1, self.hid_feats)))
        
        h = self.conv3(blocks[2], h)
        h[self.rel_list[0]] = F.tanh(self.bns[2](h[self.rel_list[0]].view(-1, self.hid_feats)))
        h[self.rel_list[1]] = F.tanh(self.bns2[2](h[self.rel_list[1]].view(-1, self.hid_feats)))
        

        h = self.conv4(blocks[3], h)


        h = {k: ((v.view(-1, self.out_feats))) for k, v in h.items()}


        return h

        # h = self.conv1(blocks[0], inputs)

        # print(blocks[0].srcdata['_ID'])
        # print('-'*40)
        # for node_t in blocks[0].srcdata['ID'].keys():
        #     print("node_t")
        #     print(blocks[0].srcdata['ID'][node_t])
        #     print(blocks[0].dstdata['ID'][node_t])
        #     print(h[node_t].shape)
        # raise NotImplementedError

        

        # self.rel_list = list(h.keys())
        # h[self.rel_list[0]] = F.leaky_relu(self.bns[0](h[self.rel_list[0]].view(-1, self.hid_feats)))
        # h[self.rel_list[1]] = F.leaky_relu(self.bns2[0](h[self.rel_list[1]].view(-1, self.hid_feats)))

        # # print("h[1]shape")
        # # for k,v in h.items():
        # #     print(k,v.shape)

        # h = self.conv2(blocks[1], h)
        # h[self.rel_list[0]]=h[self.rel_list[0]].view(-1,self.hid_feats)
        # h[self.rel_list[1]]=h[self.rel_list[1]].view(-1,self.hid_feats)
        # h = self.residual_2(h, blocks[1], blocks[2])
        # h[self.rel_list[0]] = F.leaky_relu(self.bns[1](h[self.rel_list[0]]))
        # h[self.rel_list[1]] = F.leaky_relu(self.bns2[1](h[self.rel_list[1]]))

        # # print("h[2]shape")
        # # for k,v in h.items():
        # #     print(k,v.shape)

        # h = self.conv3(blocks[2], h)
        # h[self.rel_list[0]]=h[self.rel_list[0]].view(-1,self.hid_feats)
        # h[self.rel_list[1]]=h[self.rel_list[1]].view(-1,self.hid_feats)
        # h = self.residual_2(h, blocks[2], blocks[3])
        # h[self.rel_list[0]] = F.leaky_relu(self.bns[2](h[self.rel_list[0]]))
        # h[self.rel_list[1]] = F.leaky_relu(self.bns2[2](h[self.rel_list[1]]))


        # # print("h[3]shape")
        # # for k,v in h.items():
        # #     print(k,v.shape)

        # h = self.conv4(blocks[3], h)

        # h = {k: ((v.view(-1, self.out_feats))) for k, v in h.items()}

        # # print("h[4]shape")
        # # for k,v in h.items():
        # #     print(k,v.shape)

        # # raise NotImplementedError
        # return h

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["h"] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(fn.u_dot_v("h", "h", "score"), etype=etype)
            return edge_subgraph.edata["score"]


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, etypes,args):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, etypes,args)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.rgcn(blocks, x)

        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score
