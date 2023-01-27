import functools
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from scoreNet import TimeEncoding, Dense, ScoreNet

# 基于SDE采样两个方法的特殊形式，DDPM训练似然下界，离散score是score matching和朗之万采样生成，两者可以通过sde统一

device = 'cuda'  # cuda ot cpu


def marginal_prob_std(t, sigma):
    """ 利用t和sigma计算标准差， 计算任意t时刻的扰动后条件高斯分布的标准差"""
    t = torch.tensor(t, device=device)  # 拷贝t到特定设备
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))  # 缺少数据稳定性安保措施处理


def diffusion_coeff(t, sigma):
    """计算任意t时刻的扩散系数，本例定义的SDE没有漂移系数"""
    return torch.tensor(sigma ** t, device=device)


sigma = 25.0
# 构建无参函数
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)  # 任意t时刻后加噪的标准差
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # 任意t时刻的扩散系数


# 目标函数/损失函数， reverseSDE，服从均匀函数，条件变量
# x输入,t从均匀分布中采用
def loss_fn(score_model, x, marginal_prob_std, eps=1e-5):
    """
    The loss function of training score-based generative models
    Parameters
    ----------
    score_model: A PyTorch Model instance that represents a time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    # Step 1 从[0.00001, 0.9999]中随机生成batchsize个浮点型t，
    # 相当于minmax反归一化的一个过程，不要后面的乘项也可以，乘项为了不要有完全的1和完全的0
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  # 随机*1+0,在（0,1）之间的随机数

    # Step 2 基于重参数技巧采样出分布p_t(x)的一个随机样本perturbed_x
    z = torch.randn_like(x)  # 从分布中抽出和x一样的z,保持维度
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]  # 得到扰动后的x

    # Step 3 将当前的加噪样本和时间输入到score Network中预测出分数score
    score = score_model(perturbed_x, random_t)

    # Step 4 计算score matching loss, score-条件变量计算平方差
    # 非减法而是加法，p(xt|x0)是高斯分布，求log再求梯度会变成-epslion/sigma, sigma移进去约掉
    # 对通道、长度、宽度维求和，再求均值平均到每个minibatch上
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
    return loss

# 训练时候对权重指数平滑，使得生成质量更高
class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v, in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    # EMA在每一步只更新一步新的参数， e是上一步的参数，m是新的，decay一般是0.9999缓慢更新
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
