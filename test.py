# 使用字典来查找索引
import torch
before_ids = [0,1,8,23,34,43,53]
after_ids =  [1,34,53]
id_to_index = {id_: idx for idx, id_ in enumerate(before_ids)} ### {0: 0, 1: 1, 8: 2, 23: 3, 34: 4, 43: 5, 53: 6}
indices = torch.tensor([id_to_index[id_] for id_ in after_ids], dtype=torch.long) 
### tensor([1, 4, 6])

# 获取需要相加的特征
before_features = torch.tensor([i for i in before_ids])
after_features = torch.zeros(3)
# 批量相加
after_features += before_features[indices]
print(after_features)