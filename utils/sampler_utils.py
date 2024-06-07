import torch
import dgl
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numpy.linalg import norm
from tqdm import tqdm


def sampler_train_metapath(g, path, ntype):
    """
    training MetaPath
    especially built for sampler
    """
    print('Training metapath2vec as initial author feature...')
    model = dglnn.MetaPath2Vec(g, path, window_size=3, emb_dim=512)
    dataloader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=1024,
                        shuffle=True, collate_fn=model.sample)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.025)
    for epoch in tqdm(range(10)):
        for (pos_u, pos_v, neg_v) in dataloader:
            loss = model(pos_u, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    nids = torch.LongTensor(model.local_to_global_nid[ntype])
    emb = model.node_embed(nids)
    # for fixed_version
    torch.save(emb, 'unbiased_sampler/author_feature.pt')
    return emb



def SamplerNode2Vec(args,G, path, window_size=4,epochs=150):
    device=args.device

    n2vModel = dgl.nn.pytorch.MetaPath2Vec(G,path,emb_dim=args.input_dim,window_size=window_size).to(device)

    n2vDataLoader = DataLoader(torch.arange(G.num_nodes('author')), batch_size=256, shuffle=True,collate_fn=n2vModel.sample)
    optimizer = torch.optim.SparseAdam(n2vModel.parameters(),lr=0.001)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.7,verbose=True)
    print("Start training the Node2Vec model...")
    for epoch in tqdm.tqdm(range(200)):
   
        for (pos_u, pos_v, neg_v) in n2vDataLoader:
            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)

        loss = n2vModel(pos_u, pos_v, neg_v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
        lr_sche.step()  
    with torch.no_grad():
        user_nids = torch.LongTensor(n2vModel.local_to_global_nid['author']).to(device)
        item_nids = torch.LongTensor(n2vModel.local_to_global_nid['paper']).to(device)
        node_embeddings = {}
        node_embeddings['author'] = n2vModel.node_embed(user_nids).detach().cpu()
        node_embeddings['paper'] = n2vModel.node_embed(item_nids).detach().cpu()
        print(node_embeddings['author'].shape, node_embeddings['paper'].shape)
    
    
    # after the training, to get the specific node_embedding
    nids = torch.LongTensor(n2vModel.local_to_global_nid['author'])
    emb = n2vModel.node_embed(nids)
    os.mkdir(f'unbiased_sampler/node_embedding/{args.input_dim}', exist_ok=True)
    torch.save(emb, f'unbiased_sampler/node_embedding/{args.input_dim}/author_feature.pt')
    return emb

        

# here should also passing the args
# for the args.input_dim decides the dim of the node_embedding for us to choose
def sampler_fixed_build_env(args, train_refs, cite_edges, coauthor_edges, paper_feature):
    device = args.device
    os.environ['DGLBACKEND'] = 'pytorch'

    train_ref_tensor = torch.from_numpy(train_refs.values)
    cite_tensor = torch.from_numpy(cite_edges.values)
    coauthor_tensor = torch.from_numpy(coauthor_edges.values)
    
    rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), ('author', 'coauthor', 'author'), ('paper', 'beref', 'author')]
    graph_data = {
    rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
    rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
    rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
    rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0])
    }
    hetero_graph = dgl.heterograph(graph_data)

    
    input_dim = args.input_dim
    # according the input_dim to choose the node_embeddings
    # during the initial training, we need to train the author feature and save it for future sampler use
    if not(os.path.exists(f'unbiased_sampler/node_embedding/{input_dim}/author_feature.pt')):
        # since the author_feature for specific input_dim has not been trained, we need to train it
        SamplerNode2Vec(args, hetero_graph, path=['ref', 'beref'])


    # if the node embedding has already been trained, we can directly load it
    # author_feature = torch.load('unbiased_sampler/author_feature.pt').detach()
    author_feature = torch.load(f'unbiased_sampler/node_embedding/{input_dim}/author_feature.pt').detach()

    node_features = {'author': author_feature, 'paper': paper_feature}

    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)

    return hetero_graph, rel_list



# def sampler_build_env(train_refs, cite_edges, coauthor_edges, paper_feature, device):
#     os.environ['DGLBACKEND'] = 'pytorch'
#     train_ref_tensor = torch.from_numpy(train_refs.values)
#     cite_tensor = torch.from_numpy(cite_edges.values)
#     coauthor_tensor = torch.from_numpy(coauthor_edges.values)
#     rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'), ('author', 'coauthor', 'author'), ('paper', 'beref', 'author')]
#     graph_data = {
#     rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
#     rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
#     rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
#     rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0])
#     }
#     hetero_graph = dgl.heterograph(graph_data)

#     author_feature = sampler_train_metapath(hetero_graph, ['ref', 'beref'], 'author').detach()

#     node_features = {'author': author_feature, 'paper': paper_feature}

#     hetero_graph.ndata['features'] = node_features
#     hetero_graph = hetero_graph.to(device)

#     return hetero_graph, rel_list