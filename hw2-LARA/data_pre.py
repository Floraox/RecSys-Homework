import numpy as np
import pandas as pd
import torch

# 构建物品集、用户集、属性集
# 使用原作者代码中的数据，其中positive: rating> 3，neg：rating<4

train = pd.read_csv('data/train_data.csv', header=None)
neg = pd.read_csv('data/neg_data.csv', header=None)
user_emb_matrix = pd.read_csv('data/user_emb.csv', header=None)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_attr):
        if data_attr == 'train':
            data = train

        if data_attr == 'neg':
            data = neg

        # if data_attr =='test':
        #     data = test

        self.user_col = data.loc[:, 0]
        self.item_col = data.loc[:, 1]
        self.attr_col = data.loc[:, 2]
        self.user_emb = np.array(user_emb_matrix)
        self.len = len(data)

    def __getitem__(self, index):
        user = self.user_col[index]
        item = self.item_col[index]
        user_emb = self.user_emb[user]
        attr = self.attr_col[index][1:-1].split()
        attr = np.array([int(item) for item in attr])  # str 转 int

        return user, item, attr, user_emb

    def __len__(self):
        return self.len
