import torch
import torch.nn as nn
import data_pre
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 设置超参
alpha = 0  # 正则项参数
attr_num = 18
attr_present_dim = 5  # 设置属性维度
batch_size = 1024  # {128，256，512，1024}
hidden_dim = 100  # {100，200，300，400，450}
user_emb_dim = attr_num
learning_rate = 0.0001  # {1E-05，1E-04，1E-03，1E-02}
num_epochs = 400
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 测试数据
test_item = pd.read_csv('data/test_item.csv', header=None).loc[:]
test_item = np.array(test_item)
test_attr = pd.read_csv('data/test_attribute.csv', header=None).loc[:]
test_attr = np.array(test_attr)


# 所有的nn都是sigmoid激活
# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2 * attr_num, attr_present_dim)  # attr在数据预处理时*2
        nn.init.xavier_normal_(self.G_attr_matrix.weight)
        # input:user profile
        self.l1 = nn.Linear(attr_num * attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self._initialize_weights()

    # 初始化都是Xavier,matrix、weight and bias
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_normal_(m.bias.unsqueeze(0), gain=1)  # xavier计算tensor不能fewer than 2 dimensions

    def forward(self, attribute_id):
        attr_present = self.G_attr_matrix(attribute_id)  # embedding
        z1 = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])
        z2 = torch.sigmoid(self.l1(z1))
        z3 = torch.sigmoid(self.l2(z2))
        uc = torch.sigmoid(self.l3(z3))
        return uc


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(2 * attr_num, attr_present_dim)  # attr在数据预处理时*2
        nn.init.xavier_normal_(self.D_attr_matrix.weight)
        # input:user feature + attribute
        self.l1 = nn.Linear(attr_num*attr_present_dim+user_emb_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_normal_(m.bias.unsqueeze(0), gain=1)

    def forward(self, attribute_id, user_emb):
        attr_present = self.D_attr_matrix(attribute_id)  # embedding
        feature = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])
        zd1 = torch.cat((feature, user_emb), 1).float()
        zd2 = torch.sigmoid(self.l1(zd1))
        zd3 = torch.sigmoid(self.l2(zd2))
        d_logit = self.l3(zd3)
        return d_logit
        # sigmoid(out)
        # return user attr represent以去求cosine相似度


# dataloader
train_data = data_pre.Dataset('train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
neg_data = data_pre.Dataset('neg')
neg_loader = DataLoader(neg_data, batch_size=batch_size, shuffle=True)
# initialize
generator = Generator()
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()
# optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
# loss function
criterion = nn.BCEWithLogitsLoss()

x_epoch = []  # 定义一个 x 轴的空列表用来接收动态的数据
y_p10 = []
y_ndcg10 = []
y_dloss = []
y_gloss = []
plt.ion()  # 开启一个画图的窗口


# train
for epoch in range(num_epochs):
    neg_iter = neg_loader.__iter__()
    # 1. Train D
    r = 0  # 记录取neg的次数
    for user, item, attr, user_emb in train_loader:
        optimizer_D.zero_grad()
        if r * batch_size >= neg_data.__len__():  # 读取一个batch的数据,超过总长度就break
            break
        _, _, neg_attr, neg_user_emb = neg_iter.next()
        attr = attr.long()  # 应该为LongTensor
        attr = attr.cuda()
        neg_attr = neg_attr.cuda()
        neg_user_emb = neg_user_emb.cuda()
        user_emb = user_emb.cuda()
        fake_user_emb = generator(attr)
        d_real = discriminator(attr, user_emb)
        d_fake = discriminator(attr, fake_user_emb)
        d_neg = discriminator(neg_attr.long(), neg_user_emb)

        d_real_loss = torch.mean(criterion(d_real, torch.ones_like(d_real)))
        d_fake_loss = torch.mean(criterion(d_fake, torch.zeros_like(d_fake)))
        d_neg_loss = torch.mean(criterion(d_neg, torch.zeros_like(d_neg)))
        # loss = real + fake +neg
        d_loss = d_real_loss + d_fake_loss + d_neg_loss
        d_loss.backward()
        optimizer_D.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        r = r + 1
    # 2. Train G
    for user, item, attr, user_emb in train_loader:
        optimizer_G.zero_grad()
        attr = attr.long()
        attr = attr.cuda()
        fake_user_emb = generator(attr)
        fake_user_emb = fake_user_emb.cuda()
        d_fake = discriminator(attr, fake_user_emb)
        # g_loss只和fake的有关
        g_loss = torch.mean(criterion(d_fake, torch.ones_like(d_fake)))
        g_loss.backward()
        optimizer_G.step()  # Only optimizes G's parameters
    # 3. Test
    print("Epoch {}: d_loss:{:.6f}, g_loss:{:.6f}. ".format(epoch, d_loss, g_loss), end="")
    item = torch.tensor(test_item).cuda()
    attr = torch.tensor(test_attr).long().cuda()
    g_user = generator(attr)
    result = evaluate.val(item, g_user)
    optimizer_G.zero_grad()  # 清0不然会影响d_loss

    # plot
    x_epoch.append(epoch)
    y_p10.append(result[0])
    y_ndcg10.append(result[2])
    y_dloss.append(d_loss)
    y_gloss.append(g_loss)
    plt.clf()  # 清除之前画的图

    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.plot(x_epoch, y_p10)
    plt.xlabel("Epoch")
    plt.ylabel("P@10")
    ax2 = plt.subplot(222)
    plt.plot(x_epoch, y_ndcg10)
    plt.xlabel("Epoch")
    plt.ylabel("NDCG@10")
    ax3 = plt.subplot(223)
    plt.plot(x_epoch, y_dloss)
    plt.xlabel("Epoch")
    plt.ylabel("D_loss")
    ax4 = plt.subplot(224)
    plt.plot(x_epoch, y_gloss)
    plt.xlabel("Epoch")
    plt.ylabel("G_loss")
    plt.pause(0.1)  # 暂停一秒
plt.ioff()  # 关闭画图的窗口

