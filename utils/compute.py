import numpy as np
from numpy.linalg import norm

def cos_sim(a, b):
    cos_sim = np.sum(a * b, axis = 1) / (norm(a, axis = 1) * norm(b, axis = 1))
    return cos_sim

def compute_loss(pos_score, neg_score, etype):
    n_edges = pos_score[etype].shape[0]
    return (1 - pos_score[etype].unsqueeze(1) + neg_score[etype].view(n_edges, -1)).clamp(min=0).mean()