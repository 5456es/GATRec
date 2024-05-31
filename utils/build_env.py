from .node2vec import Node2Vec
import os
import torch
import dgl

    
def build_env(args,train_refs, cite_edges, coauthor_edges, paper_feature):
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

    author_feature , paper_feature_= Node2Vec(args,hetero_graph, ['coauthor','ref','cite','beref'])

    node_features = {'author': author_feature, 'paper': paper_feature if args.input_dim == 512 else paper_feature_}

    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)

    return hetero_graph, rel_list