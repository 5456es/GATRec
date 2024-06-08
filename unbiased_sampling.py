import torch
import dgl
import numpy as np
from utils import cos_sim
import torch
from utils import sampler_fixed_build_env, process_data, cos_sim
from model import Model


node_embeddings = None


def cos_sim_2nd_sampler(author_idx_tensor, paper_idx_tensor, sample_num):
    _author_idx = author_idx_tensor.cpu()
    _paper_idx = paper_idx_tensor.cpu()
    res = cos_sim(np.array(node_embeddings['author'][_author_idx]), 
                  np.array(node_embeddings['paper'][_paper_idx]))
    _prob = np.ones(res.shape)

    # 几乎调成为 0.
    _prob[res > 0.55] = 0.01
    _prob = _prob / _prob.sum()
    _sample_idx = np.random.choice(np.arange(len(res)), size=sample_num, replace=False, p=_prob)
    return _sample_idx
     

    
def load_model_get_sampler(args):

    global node_embeddings
    # node_embeddings = torch.load('unbiased_sampler/node_embedding/512/2024-06-07 14:04:44_0.9417910995340646_node_feature.pth')
    node_embeddings = torch.load(args.unbiased_sampler_node_feature_path)
    print('Load node features successfully!')
    class NewSampler(dgl.dataloading.negative_sampler._BaseNegativeSampler):
        def __init__(self, k, exclude_self_loops=True, replace=False):
            self.k = k
            self.exclude_self_loops = exclude_self_loops
            self.replace = replace

        def _generate(self, g, eids, canonical_etype):
            if canonical_etype == ('author', 'ref', 'paper') or canonical_etype == ('paper', 'beref', 'author'):
                _first_sample_res =  g.global_uniform_negative_sampling(
                    2 * len(eids) * self.k,
                    self.exclude_self_loops,
                    self.replace,
                    canonical_etype,
                )
                if canonical_etype == ('author', 'ref', 'paper'):
                    _sample_idx = cos_sim_2nd_sampler(_first_sample_res[0], _first_sample_res[1], len(eids) * self.k)
                elif canonical_etype == ('paper', 'beref', 'author'):
                    _sample_idx = cos_sim_2nd_sampler(_first_sample_res[1], _first_sample_res[0], len(eids) * self.k)
                return _first_sample_res[0][_sample_idx], _first_sample_res[1][_sample_idx]
            else:
                return g.global_uniform_negative_sampling(
                    len(eids) * self.k,
                    self.exclude_self_loops,
                    self.replace,
                    canonical_etype,
                )
    return NewSampler

