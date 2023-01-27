# @ define a UNet time-dependant SN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import array, argmax
import functools

# -----------------Embed time and label condition--------------#
# Score Net 不依赖加噪，输入x和t#
class TimeEncoding(nn.Module):
    """时间傅里叶编码, SDE更一般形式，有限时间不包括,高斯随机特征编码"""

    def __init__(self, embed_dim, scale=30.):
        # scale 从高斯分布中随机生成 W，不可学习的
        super().__init__()
        # randomly sample weights during initialization. These weights are fixed
        # during optimization are not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
    # def forward(self, x, timesteps, y):
        # 时间t进入, sin和cos二维拼接，类似transformer position encoding，这里是time encoding
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)


# 扩维MLP
class Dense(nn.Module):
    """A fully connected layer that reshape outputs to feature maps"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 扩充维度
        return self.dense(x)[..., None, None]  # 对输出扩充了两个维度


class ScoreNet(nn.Module):
    """基于Unet的时间依赖的分数估计模型"""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, num_classes=None):
        """ Initialize a time-dependant score-based network
        Args:
        Parameters
        ----------
        marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel_p{x0}(x(t)|x(0))
        channels: The number of channels for feature maps of each resolution
        embed_dim: The dimensionality of Gaussian random features embeddings
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        # 时间编码层，embed时间编码加linear特征变换

        self.num_classes = num_classes
        self.time_embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))

        # YY: add class
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Unet的编码器部分，空间不断减小，通道不断增大
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)  # to skip connection
        # 将对应编码器部分链接到解码器部分（第一部分的像素接到解码器部分）, dense用于引入时间
        self.dense1 = Dense(embed_dim, channels[0])  # 引入时间
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Unet解码器，反卷积，空间不断增大，通道不断减小，并且有来自编码器部分的skip connection
        # kernel size, padding,stride设置一样--上采样，多少倍通过公式计算
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # 最终输出，最后一个反卷积
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 3, 3, stride=1)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y=None):
        # print(t)
        # print(y)

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # 对时间进行编码，前向送入t
        embed = self.act(self.time_embed(t))  # 时间信息
        # print(x.shape, t.shape, y.shape, embed.shape)
        # print(self.label_emb(y).shape)
        # print(embed)
        if self.num_classes is not None:
            # print(x.shape, y.shape, t.shape)
            # assert y.shape[0] == (x.shape[0],)
            embed = embed + self.label_emb(y)

        # print(embed.shape)

        # 编码器部分前向计算, 空间降维，通道扩维
        h1 = self.conv1(x)
        h1 += self.dense1(embed)  # 注入时间t, 不同的dense层做后处理
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        # print(h1.shape)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)  # 注入时间t
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        # print(h2.shape)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)  # 注入时间t
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        # print(h3.shape)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)  # 注入时间t
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # print(h4.shape)

        # 解码器部分前向计算
        h = self.tconv4(h4)
        h += self.dense5(embed)  # 注入时间t
        h = self.tgnorm4(h)
        h = self.act(h)
        # print(h.shape, h3.shape)
        # assert 1==0
        h = self.tconv3(torch.cat((h, h3), dim=1))  # skip connection
        h += self.dense6(embed)  # 注入时间t
        h = self.tgnorm3(h)
        h = self.act(h)
        # print(h.shape)
        # print(h.shape, h2.shape)
        h = self.tconv2(torch.cat((h, h2), dim=1))  # skip connection
        h += self.dense7(embed)  # 注入时间t
        h = self.tgnorm2(h)
        h = self.act(h)
        # print(h.shape, h1.shape)
        h = self.tconv1(torch.cat((h, h1), dim=1))  # skip connection
        # print(h.shape)
        # Normalize output 除以二阶范数的平方的平方根倒数,目的是希望预测的分数的二阶范数逼近于真实分数的二阶范数
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        # 除以二范数的期望值，相当于把lambda移到了scoreNet里
        # print(h.shape)
        return h