# @title Training (double click to expand or collapse)

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import tqdm
from torchvision.datasets import CIFAR10
import os
from scoreNet import ScoreNet
from sdeNoise import marginal_prob_std, marginal_prob_std_fn, device, EMA, loss_fn
from torch.utils.tensorboard import SummaryWriter

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs = 50  # @param {'tpye:'integer'} 学习率, loss值大概在16
## size of  a mini-batch
batch_size = 32  # @param {'type':'integer'}
## learning rate
lr = 1e-4  # @param {'type':'number'}

# MNIST 60000个训练集和10000个测试集构成的28*28灰度图
dataset = CIFAR10('.', train=True, transform=transform.ToTensor(), download=True)  # 变成浮点型
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(score_model.parameters(), lr=lr)
# tqdm_epoch = tqdm.tqdm(range(n_epochs))  # 进度条，循环

ema = EMA(score_model)

resume = True  # 设置是否需要从上次的状态继续训练
if resume:
    if os.path.isfile("checkpoint2/ckpt_49.pth"):
        print("Resume from checkpoint2...")
        path_checkpoint = "checkpoint2/ckpt_49.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cuda'))
        # score_model.load_state_dict(checkpoint_3['score_model.state_dict'])
        # optimizer.load_state_dict(checkpoint_3['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("====>loaded checkpoint2 (epoch{})".format(checkpoint['epoch']))
        avg_loss = checkpoint['avg_loss']
        num_items = checkpoint['num_items']
    else:
        print("====>no checkpoint_3 found.")
        start_epoch = 1  # 如果没进行训练过，初始训练epoch值为1
        avg_loss = 0
        num_items = 0

for epoch in range(start_epoch, n_epochs):
    # print(epoch in tqdm_epoch)
    for x, y in data_loader:  # 无条件建模，未用上y标签 0~9
        # avg_loss = 0
        # num_items = 0
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)  # 传入x计算标准差计算分数匹配的loss，做参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(score_model)
        # if isinstance(score_model, torch.nn.DataParallel):
        #     score_model = score_model.module
        ema.update(score_model)  # 更新平滑模型
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    print('Epoch:', epoch, ' || ',
          'Avg_loss:', avg_loss, ' || ',
          'Num_items', num_items, ' || '
                                  'Average ScoreMatching Loss: {:5f}'.format(avg_loss / num_items))

    writer = SummaryWriter('log')
    writer.add_scalar("Average_scoreMatching_Loss", round((avg_loss / num_items), 5), epoch)
    # if epoch % 5 == 1 or epoch == 49:
    torch.save({
        'epoch': epoch,
        'model_state_dict': score_model.state_dict(),
        'avg_loss': avg_loss,
        'num_items': num_items
    }, f'checkpoint2/ckpt_{epoch}.pth')
    # torch.save(score_model.state_dict(), f'./checkpoint_3/ckpt_{epoch}.pth')  # fstring保存每一个epoch的checkpoint
