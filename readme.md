# GATRec

## Unbiased Sampler

Unbiased Sampler is guided by a pretrained-well model. The pretrained well then to guide the GlobalUniformSampler to sample, guided by the well-pretrained model, the Unbiased Sampler can reduce the probability of sample those edges with high probobility seen as positive samples in the unlabelled data.

To train the GATRec with the Unbiased Sampler from scratch, you need to 
1. first train a normal model to constitude the Unbiased sampler
2. based on the normal model, train a better model with Unbiased sampler.