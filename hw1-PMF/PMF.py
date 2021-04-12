import numpy as np
import copy
import matplotlib.pyplot as plt

# 处理数据，id转为index，获得评过分的user和movie的数量
user_set = dict()
movie_set = dict()
data = []
num_user, num_movie = 0, 0

with open("./ratings.dat") as f:
    for line in f.readlines():
        user, movie, rating, time = line.split('::')
        if user not in user_set:
            user_set[user] = num_user  # index并计数
            num_user += 1
        if movie not in movie_set:
            movie_set[movie] = num_movie
            num_movie += 1
        data.append([user_set[user], movie_set[movie], float(rating)])
f.close()

# 划分数据
ratio = 0.8
data = np.array(data)
np.random.shuffle(data)
train_data = data[:int(len(data) * ratio)]
test_data = data[int(len(data) * ratio):]


def RMSE(pred, rate):
    return np.sqrt(np.mean(np.square(pred-rate)))


class PMF():
    def __init__(self, R, lambda_u, lambda_v, D, lr, iters):
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.R = R
        self.iterations = iters
        self.lr = lr
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1  # I：indicator function
        self.U = np.random.normal(0, 0.1, (np.size(R, 0), D))  # 初始化U
        self.V = np.random.normal(0, 0.1, (np.size(R, 1), D))  # 初始化V

    def train(self, data=None):
        # 保存loss和rmse的值
        loss_list = []
        rmse_list = []
        tmp_rmse = None

        for iter in range(self.iterations):
            # 偏导
            grads_u = np.dot(self.I * (self.R - np.dot(self.U, self.V.T)), -self.V) + self.lambda_u * self.U
            grads_v = np.dot((self.I * (self.R - np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_v * self.V
            # 梯度更新
            self.U = self.U - self.lr * grads_u
            self.V = self.V - self.lr * grads_v
            # 计算loss
            loss = self.loss()
            loss_list.append(loss)
            # 计算rmse
            preds = self.predict(data)
            rmse = RMSE(data[:, 2], preds)
            rmse_list.append(rmse)
            print('iteration:{:d} ,loss:{:f}, rmse:{:f}'.format(iter, loss, rmse))

            if tmp_rmse and (tmp_rmse - rmse) <= 0:
                print('iterations:{: d}收敛'.format(iter))
                break
            else:
                tmp_rmse = rmse

        return self.U, self.V, loss_list, rmse_list

    # 损失函数
    def loss(self):
        loss = np.sum(self.I * (self.R - np.dot(self.U, self.V.T)) ** 2) + self.lambda_u * np.sum(np.square(self.U)) + self.lambda_v * np.sum(np.square(self.V))
        return loss

    def predict(self, data):
        index_data = np.array([[int(e[0]), int(e[1])] for e in data], dtype=int)
        u = self.U.take(index_data.take(0, axis=1), axis=0)  # 取对应index处的值进行运算
        v = self.V.take(index_data.take(1, axis=1), axis=0)
        preds = np.sum(u * v, 1)  # 获得预测的评分矩阵
        return preds


# 设定相关参数
R = np.zeros([num_user, num_movie])
for each in train_data:
    R[int(each[0]), int(each[1])] = float(each[2])  # 由train_data生成R
lambda_u = 0.01
lambda_v = 0.01
D = 15
lr = 0.0001
iters = 500

# 训练模型
model = PMF(R=R, lambda_u=lambda_u, lambda_v=lambda_v, D=D, lr=lr, iters=iters)
U, V, loss_list, rmse_list = model.train(data=train_data)
preds = model.predict(data=test_data)
test_rmse = RMSE(preds, test_data[:, 2])
print('test rmse:{:f}'.format(test_rmse))

# 画图
# range(len(rmse_list))
plt.plot(rmse_list)
plt.title('Rmse Curve')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
# plt.grid()
plt.show()


# todo: 归一化让loss值小一些
