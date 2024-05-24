import torch
import numpy as np
import random
import dgl
import argparse

from utils import *
from train import train


def parse_arguments():
    parser = argparse.ArgumentParser("The main entry point for the GNN Academic Paper Link Prediction program")
    ### Data Loading and Saving
    parser.add_argument("--data_path", type=str,default="./data/", help="The path to the data folder")
    parser.add_argument("--save_path", type=str,default='./save/', help="The path to save the model")
    parser.add_argument("--load_path", type=str,default='./save/', help="The path to load the model")
    parser.add_argument("--save_log_path", type=str,default='./logs/', help="The path to save the log")

    ### Train Parameters
    parser.add_argument("--input_dim", type=int, default=512, help="The input dimension for the model")
    parser.add_argument("--hidden_dim", type=int, default=512, help="The hidden dimension for the model")
    parser.add_argument("--output_dim", type=int, default=256, help="The output dimension for the model")
    parser.add_argument("--batch_size", type=int, default=10000, help="The batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="The number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0002, help="The learning rate for training")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="The learning rate decay for training")
    parser.add_argument('--lr_period', default=10, type=int, help='period for lr_scheduler')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="The weight decay for training")
    parser.add_argument("--k", type=int, default=5, help="The number of negative samples to use")

    ### Other Parameters
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu", help="The device to run the model on, [cuda:x] or cpu")
    parser.add_argument("--seed", type=int, default=41, help="The seed for the random number generator")
    parser.add_argument("--log_interval", type=int, default=100, help="The interval to log the training results")

    return parser.parse_args()

def main(args):
    ### Set and Preprocess 
    set_seed(args.seed)
    if args.device == "cuda:1":
        torch.cuda.set_device(1)
        print(torch.cuda.get_device_name())
        print(torch.cuda.current_device())
    train_refs, test_refs, refs_to_pred, cite_edges, coauthor_edges, paper_feature=process_data(args)
    hetero_graph, rel_list = build_env(args,train_refs, cite_edges, coauthor_edges, paper_feature)
    best_embed, best_thresh, best_f1 = train(args, hetero_graph, test_refs, rel_list)
    gen_csv_prediction(best_embed, refs_to_pred, best_thresh, best_f1, args.seed)
    
   
    ### To be implemented

if __name__ == "__main__":

    args = parse_arguments()
    print("Arguments: \n"+"-"*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-" * 50)

    torch.cuda.empty_cache()
    
    main(args)