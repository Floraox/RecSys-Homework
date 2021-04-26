import numpy as np
import pandas as pd
import torch


user_attribute_matrix = pd.read_csv('data/user_attribute.csv', header=None)
user_attribute_matrix = torch.tensor(np.array(user_attribute_matrix[:]), dtype=torch.float)
ui_matrix = pd.read_csv('data/ui_matrix.csv', header=None)
ui_matrix = np.array(ui_matrix[:])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 给出新的物品及其属性，生成可能会喜欢它的用户（user_attribute)
# 找到top k用户
def get_similar_user(g_fake_user, k):
    g_fake_user = g_fake_user.to(device)
    user_embed_matrix = user_attribute_matrix.to(device)
    similar_matrix = torch.matmul(g_fake_user, user_embed_matrix.T)  # cosine相似
    index = torch.argsort(-similar_matrix)  # 选择内积大的k个
    torch.set_printoptions(profile="full")
    return index[:, 0:k]


def dcg_at_k(r, k):
    r = np.asfarray(r)
    return np.sum((np.power(2, r)-1)/(np.log2(np.arange(2, r.size+2))))


# 推荐排序在对的位置有多少
def ndcg_at_k(item, g_fake_user, k):
    similar_user = get_similar_user(g_fake_user, k)
    sum = 0.0
    for test_item, user_list in zip(item, similar_user):
        r = []
        for user in user_list:
            r.append(ui_matrix[user][test_item])  # 找到对应的item是否感兴趣
        r_ideal = sorted(r, reverse=True)
        idcg = dcg_at_k(r_ideal, k)
        if idcg == 0:
            sum += 0
        else:
            sum += (dcg_at_k(r, k)/idcg)

    return sum/item.__len__()


# 计算推荐对了多少
def p_at_k(item, g_fake_user, k):
    similar_user = get_similar_user(g_fake_user, k)
    count = 0
    total = item.__len__()
    for item, user_list in zip(item, similar_user):
        for user in user_list:
            if ui_matrix[user, item] == 1:
                count += 1
    return count / (total * k)


def val(item, g_fake_user):
    # similar_user = get_similar_user(g_fake_user, 10)
    p10 = p_at_k(item, g_fake_user, 10)
    p20 = p_at_k(item, g_fake_user, 20)
    ndcg_10 = ndcg_at_k(item, g_fake_user, 10)
    ndcg_20 = ndcg_at_k(item, g_fake_user, 20)
#    print("p@10:%.4f, p@20:%.4f, ndcg@10:%.4f, ndcg@20:%.4f" % (p10, p20, ndcg_10, ndcg_20))
    print("p@10:%.4f, ndcg@10:%.4f" % (p10, ndcg_10))
    result = [p10, p20, ndcg_10, ndcg_20]
    return result

# todo: M_at_k
