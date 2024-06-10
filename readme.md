# GATRec
This is a GAT-based model to do recommendation on the academic network.

## Quick Start
```shell
 python main.py --data_path --batch_size 10000 --lr 0.0001 --lr_end 0.00001 --num_epochs 100 
```
This command will train a GATRec model with the default setting.

### Use pretrained model
```shell
 python main.py --load_path path_to_model
```
Attention: Provided model follows the default setting, if you want to use the pretrained model, please make sure the model is trained with the same setting.
### Use pretrained node feature
```shell
 python main.py --load_embed_path path_to_node_feature
```
You may use our pretrained node feature to train the model or you can train the model with your own node feature by setting load_embed_path to None, the pretrained node feature shall be automatically saved in the ./embedding folder.




### Unbiased Sampler

Unbiased Sampler is guided by a pretrained-well model. The pretrained well then to guide the GlobalUniformSampler to sample, guided by the well-pretrained model, the Unbiased Sampler can reduce the probability of sample those edges with high probobility seen as positive samples in the unlabelled data.

#### Train GATRec with provided Data

```shell
 python main.py --unbiased_sampler True --unbiased_sampler_node_feature_path 'unbiased_sampler/node_embedding/512/2024-06-07 14:04:44_0.9417910995340646_node_feature.pth'
```

### Train GATRec From Scratch

To train the GATRec with the Unbiased Sampler from scratch, you need to 
1. first train a normal model and use the node embedding to pass the model to get the ==node feature==(which can be used to calculate the cosine similarity between nodes).
2. based on the pretrained node feature, train the Unbiased Sampler model.

The specific steps are as follows:
1. Train a normal model
```shell
python main.py --unbiased_sampler False --save_node_features True
```

2. Use the unbiased Sampler to guide your new train.
Find the path of the former trained node feature, it usually locates in the `unbiased_sampler/node_embedding/{arg.input_dim}/{arg.time}_{arg.acc}_node_feature.pth`, and then use the following command to train the model.

```shell
 python main.py --unbiased_sampler True --unbiased_sampler_node_feature_path unbiased_sampler/node_embedding/{arg.input_dim}/{arg.time}_{arg.acc}_node_feature.pth
```

### Residual GATRec

```shell
 python main.py --data_path --batch_size 10000 --lr 0.0001 --lr_end 0.00001 --num_epochs 100 --hidden_dim 512 --residual True
```

Attention:  The residual needs hidden_dim to be the same as the input_dim. And due to our poor implementation, the residual model is very time-consuming. 