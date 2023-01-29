代码详解
====

SBGM_CIFAR10
------
Score based generative model trained with cifar10, scoreMatching loss略微大

SGMNIST
------
Score based generative model trained with MINST， scoreMatching loss差不多

classifier
------
score based generative classifier, based on Yang Song(42), the SDE one, combined 43 the maximum likelihood.
scoreNet.py : the model modified with Unet
sdeNoise.py: sdeNoise added the same as (42)
sbCIFAR10.py: model trained
sample.py : sample with 3 methods including ODE
likelihood.py: the nll calculated

RESULTS: nll is about 6.11, 训练收敛结果不理想，收敛不到位，平均scoreMatching loss 是122

