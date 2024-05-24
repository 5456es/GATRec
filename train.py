import torch
import dgl
import numpy as np
from model import Model
import time
from sklearn.metrics import precision_recall_fscore_support
from utils import *
from tqdm import tqdm
import os


def train(args, hetero_graph, test_refs, rel_list):
    device=args.device
    train_eid_dict = {
        etype: hetero_graph.edges(etype=etype, form='eid')
        for etype in hetero_graph.etypes}

    sampler = dgl.dataloading.NeighborSampler([4] *4 )
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(4))
    dataloader = dgl.dataloading.DataLoader(
        hetero_graph,                                 
        train_eid_dict,  
        sampler,                              
        device=device,                          
        batch_size=args.batch_size,    
        shuffle=True,       
        drop_last=False,    
        num_workers=0,     
    )

    model = Model(args.input_dim, args.hidden_dim, args.output_dim, rel_list).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

    model.load_state_dict(torch.load("./save/1716286987.236977_0.9349142949785553_model.pth"))

    best_f1 = 0
    best_thresh = None
    best_embed = None

    epoch_loss_log=[]
    print(f'Start training GAT on {device}...')
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = []
        t1 = time.time()
        print(f'epoch {epoch + 1}/{args.num_epochs}, ', end='\n')
        for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tqdm(dataloader)):
            # print(input_nodes)
            # for key in input_nodes:
            #     print(input_nodes[key].shape)
            # print(blocks[0].srcdata)
            # print(positive_graph)
            # print(negative_graph)
            
            input_features = blocks[0].srcdata['features']
            # for i in blocks[0].srcdata:
            #     print(i)
            #     print(blocks[0].srcdata[i])

            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            loss = compute_loss(pos_score, neg_score, rel_list[-1])
            if step % 50 == 0:
                print(f'loss: {loss.item():.5f}')

            epoch_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        lr_sche.step()
        print(f'loss: {(np.array(epoch_loss).mean()):.5f}')
        epoch_loss_log.append(np.array(epoch_loss).mean())
        model.eval()
        with torch.no_grad():
            blocks = [dgl.to_block(hetero_graph) for _ in range(4)]
            node_embeddings = model.rgcn(blocks, hetero_graph.ndata['features'])
        node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}

        test_arr = np.array(test_refs.values)
        res = cos_sim(np.array(node_embeddings['author'][test_arr[:, 0]]), np.array(node_embeddings['paper'][test_arr[:, 1]]))

        # Generate predict labels
        lbl_true = test_refs.label.to_numpy()
        lbl_true = lbl_true.flatten()
        lbl_pred = np.array(res)
        median = np.median(lbl_pred)

        cur_best_f1 = 0
        for i in range (-50, 50):
            threshold = best_thresh + 0.001 * i if best_thresh is not None else median + 0.001 * i
            lbl_pred_copy = lbl_pred.copy()
            lbl_pred_copy[lbl_pred >= threshold] = 1
            lbl_pred_copy[lbl_pred < threshold] = 0
            # Test scores
            precision, recall, f1, _ = precision_recall_fscore_support(lbl_true, lbl_pred_copy, average='binary')
            if f1 > cur_best_f1:
                cur_pr = precision
                cur_re = recall
                cur_best_f1 = f1
                cur_best_thresh = threshold
                cur_best_i = i
                acc = (lbl_true == lbl_pred_copy).mean()
        if cur_best_f1 > best_f1:
            print('NEW BEST PREDICTION!')
            best_f1 = cur_best_f1
            best_embed = node_embeddings
            best_thresh = cur_best_thresh
        
        t2 = time.time()
        print(f"acc:{acc:.4f}, precision:{cur_pr:.4f}, recall:{cur_re:.4f}, F1-Score: {cur_best_f1:.5f}, best: {best_f1:.5f}, time: {(t2-t1):.2f}s")

    ### save the epoch loss log as {time}_loss.npy
    save_log_path = args.save_log_path
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    np.save(f'{save_log_path}/{time.time()}_{best_f1}_loss.npy', np.array(epoch_loss_log))

    ### save the model as {time}_model.pth
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), f'{save_path}/{time.time()}_{best_f1}_model.pth')

    return best_embed, best_thresh, best_f1