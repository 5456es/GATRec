import pickle 
import pandas as pd

cite_file = "paper_file_ann.txt"
train_ref_file = "bipartite_train_ann.txt"
test_ref_file = "bipartite_test_ann.txt"
coauthor_file = "author_file_ann.txt"
feature_file = "feature.pkl"


def read_txt(file):
    res_list = list()
    with open(file, "r") as f:
        line_list = f.readlines()
    for line in line_list:
        res_list.append(list(map(int, line.strip().split(' '))))      
    return res_list


def process_data(args):
    data_path = args.data_path
    citation = read_txt(data_path + cite_file)
    existing_refs = read_txt(data_path + train_ref_file)
    refs_to_pred = read_txt(data_path + test_ref_file)
    coauthor = read_txt(data_path + coauthor_file)
    with open(data_path+feature_file, 'rb') as f:
      paper_feature = pickle.load(f)


    
    print(
    "Number of citation edges: {}\n\
    Number of existing references: {}\n\
    Number of author-paper pairs to predict: {}\n\
    Number of coauthor edges: {}\n\
    Shape of paper features: {}"
    .format(len(citation), len(existing_refs), len(refs_to_pred), len(coauthor), paper_feature.shape))
    
    cite_edges = pd.DataFrame(citation, columns=['source', 'target'])
    cite_edges = cite_edges.set_index(
        "c-" + cite_edges.index.astype(str)
    )

    ref_edges = pd.DataFrame(existing_refs, columns=['source', 'target'])
    ref_edges = ref_edges.set_index(
        "r-" + ref_edges.index.astype(str)
    )

    coauthor_edges = pd.DataFrame(coauthor, columns=['source', 'target'])
    coauthor_edges = coauthor_edges.set_index(
        "a-" + coauthor_edges.index.astype(str)
    )

    node_tmp = pd.concat([cite_edges.loc[:, 'source'], cite_edges.loc[:, 'target'], ref_edges.loc[:, 'target']])
    node_papers = pd.DataFrame(index=pd.unique(node_tmp))

    node_tmp = pd.concat([ref_edges['source'], coauthor_edges['source'], coauthor_edges['target']])
    node_authors = pd.DataFrame(index=pd.unique(node_tmp))

    train_refs = ref_edges.sample(frac=0.9,random_state=0,axis=0)
    test_true_refs = ref_edges[~ref_edges.index.isin(train_refs.index)]
    test_true_refs.loc[:, 'label'] = 1

    false_source = node_authors.sample(frac=test_true_refs.shape[0]/node_authors.shape[0],random_state=0,replace=True,axis=0)
    false_target = node_papers.sample(frac=test_true_refs.shape[0]/node_papers.shape[0],random_state=0,replace=True,axis=0)
    false_source = false_source.reset_index()
    false_target = false_target.reset_index()
    test_false_refs = pd.concat([false_source, false_target], axis=1)
    test_false_refs.columns = ['source', 'target']
    test_false_refs = test_false_refs[test_false_refs.isin(ref_edges) == False]
    test_false_refs.loc[:, 'label'] = 0

    test_refs = pd.concat([test_true_refs, test_false_refs.iloc[:min(len(false_source), len(false_target))]])
    test_refs = test_refs.sample(frac=1,random_state=0,axis=0)

    return train_refs, test_refs, refs_to_pred, cite_edges, coauthor_edges, paper_feature