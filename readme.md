# GATRec

## Unbiased Sampler

Unbiased Sampler is guided by a pretrained-well model. The pretrained well then to guide the GlobalUniformSampler to sample, guided by the well-pretrained model, the Unbiased Sampler can reduce the probability of sample those edges with high probobility seen as positive samples in the unlabelled data.

### Train GATRec with provided Data

```shell
python main.py --unbiased_sampler True --unbiased_sampler_guide_model_path 'unbiased_sampler/node_embedding/512/2024-06-07 14:04:44_0.9417910995340646_model.pth'
```

### Train GATRec From Scratch

To train the GATRec with the Unbiased Sampler from scratch, you need to 
1. first train a normal model to constitude the Unbiased sampler
2. based on the normal model, train a better model with Unbiased sampler.

The specific steps are as follows:
1. Train a normal model
```shell
python main.py --unbiased_sampler False --save_node_embeddings True
```

You can freely add other feasible parameters to the command line to train the sampler guide model.

After training, you would get your newest model path 'guide_model_path' in the `./save' directory, which is usually the one with the latest timestamp in the model's name.

2. Record your first-step model parameters in `unbiased_sampler.py` class NewSamplerModelArgs.

3. Use the unbiased Sampler to guide your new train.
```shell
python main.py --unbiased_sampler True --unbiased_sampler_guide_model_path guide_model_path
```
.