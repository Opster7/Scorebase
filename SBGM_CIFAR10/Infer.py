"""
导入训练好的MNIST模型并对比不同的采样算法
"""
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time
import torch
import numpy as np
from sample import euler_sample
from sdeNoise import marginal_prob_std_fn, diffusion_coeff_fn

# Load the pre-trained checkpoint from disk
from sbCIFAR10 import score_model

device = 'cuda'
ckpt = torch.load('./checkpointC/ckpt_49.pth',map_location=device) # load path 更改
score_model.load_state_dict(ckpt['model_state_dict'])

sample_batch_size = 64 #采样64个样本
sampler = euler_sample  #三种任选

t1 = time.time()
# Generate samples using the specified sampler
samples = sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, sample_batch_size, device = device)
t2 = time.time()
print(f"{str(sampler)}采样耗时{t2-t1}s")

# Sample visualization
samples = samples.clamp(0.0,1.0) # 截断，只保留0~1之间的数
plt.show()
#jpy matplotlib inline
sample_grid = make_grid(samples, nrow = int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
# permute转置将通道维转到最后一维
plt.imshow(sample_grid.permute(1,2,0).cpu(),vmin=0,vmax=1.) # cpu()/cuda()
plt.show()


