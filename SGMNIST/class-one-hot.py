import torch
import torchvision
from matplotlib import transforms
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def cifar10():
    # 下载CIFAR-10数据集到当前data文件夹中
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    # 从本地硬盘上读取一条数据 (包括1张图像及其对应的标签)
    image, label = train_dataset[0]
    print(image.size())  # 输出 torch.Size([3, 32, 32])
    print(label)  # 输出 6

    # 数据加载准备 (开启数据加载的线程和队列).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,  # 该参数表示每次读取的批样本个数
                                               shuffle=True)  # 该参数表示读取时是否打乱样本顺序
    # 创建迭代器
    data_iter = iter(train_loader)
    # 当迭代开始时, 队列和线程开始读取数据
    images, labels = data_iter.next()

    # print(images.size())  # 输出 torch.Size([64, 3, 32, 32])
    # print(labels.size())  # 输出 torch.Size([64])

    # 实际使用时使用下面的方式读取每一批（batch）样本
    for images, labels in train_loader:
        # 在此处添加训练代码
        pass

    # digit = train_loader.dataset.data[1]
    # plt.imshow(digit,cmap=plt.cm.binary)
    # plt.show()
    # print(classes[train_loader.dataset.targets[1]])


# label - onehot
# def test():
#     batch_size = 8
#     class_num = 10
#     label = np.random.randint(0, class_num, size=(batch_size, 1))
#     label = torch.LongTensor(label)
#     y_one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
#     print(y_one_hot)


if __name__ == "__main__":
    # onehot = Lambda(lambda y:torch.zeros(10,dtype = torch.float).scatter_(dim=0,index = torch.tensor(y),value = 1)) # 创建10维度0向量
    # a = torch.arange(10).reshape(2, 5).float()
    # b = torch.zeros(3, 5)
    # b_ = b.scatter(dim=0, index=torch.LongTensor([[1, 2, 1, 1, 2], [2, 0, 2, 1, 0]]), src=a)
    # y = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
    # scatter_(dim=0,index = torch.tensor(y),value = 1)
    # labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # n_way = 10
    # n_shot = 1
    # print(labels.shape)  # torch.Size([5])
    # one_hot_labels = torch.zeros(n_way * n_shot, n_way).scatter_(1, labels.view(-1, 1), 1)
    # print(one_hot_labels)

    # define example
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
    values = array(classes)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)
