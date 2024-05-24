import random
import dgl
from dgl.nn.pytorch import MetaPath2Vec
import torch
from torch.utils.data import DataLoader
import tqdm

def Node2Vec(G, path, window_size=3,epochs=100):
    device="cuda:1" if torch.cuda.is_available() else "cpu"

    n2vModel = dgl.nn.pytorch.MetaPath2Vec(G,path,emb_dim=512,window_size=window_size).to(device)
    # if paper_feature!=None:
    #     with torch.no_grad():
    #         try:
    #             print(n2vModel.node_embed(0))
    #         except:
    #             print(n2vModel.node_embed)

    n2vDataLoader = DataLoader(torch.arange(G.num_nodes('author')), batch_size=512, shuffle=True,collate_fn=n2vModel.sample)
    optimizer = torch.optim.SparseAdam(n2vModel.parameters(),lr=0.01)

    print("Start training the Node2Vec model...")
    for epoch in range(50):
        with tqdm.tqdm(n2vDataLoader) as tq:
            for (pos_u, pos_v, neg_v) in tq:
                pos_u = pos_u.to(device)
                pos_v = pos_v.to(device)
                neg_v = neg_v.to(device)

            loss = n2vModel(pos_u, pos_v, neg_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
                            
    with torch.no_grad():
        user_nids = torch.LongTensor(n2vModel.local_to_global_nid['author']).to(device)
        item_nids = torch.LongTensor(n2vModel.local_to_global_nid['paper']).to(device)
        node_embeddings = {}
        node_embeddings['author'] = n2vModel.node_embed(user_nids).detach().cpu()
        node_embeddings['paper'] = n2vModel.node_embed(item_nids).detach().cpu()
        print(node_embeddings['author'].shape, node_embeddings['paper'].shape)
    return node_embeddings['author']
        