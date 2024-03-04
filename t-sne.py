import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_with_tsne(features, labels, save_path, seed=42):
    # 进行特征降维和映射
    tsne = TSNE(n_components=2,random_state=seed) #4
    features_tsne = tsne.fit_transform(features)

    # 定义标签类别名称和颜色
    label_names = {
        0: "support",
        1: "refute",
        2: "neutral"
    }
    # colors = {
    #     0: "red",
    #     1: "blue",
    #     2: "green"
    # }
    colors = {
        0: "green",
        1: "red",
        2: "blue"
    }
    # colors = {
    #     0: "#12E666",
    #     1: "#F42208",
    #     2: "#EBE512"
    # }
    # 绘制散点图
    unique_labels = np.unique(labels)

    plt.figure(figsize=(6, 4))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=label_names[label], c=colors[label],s=4)

    plt.legend()
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)  # 去掉刻度
    plt.savefig(save_path, transparent=True, dpi=300)
    plt.show()
    
# features = np.random.randn(1000, 10)  # 假设有1000个样本，每个样本有10个特征
# labels = np.random.randint(0, 3, 1000)  # 假设有3个标签，取值范围为0到2

# features = np.load('all_lf.npy')
# features = features[:250]
# bs = features.shape[0]
# features = np.reshape(features, (-1,128))
# print(features.shape)
# labels = np.array([0,1,2]*bs)
# print(labels.shape)

# seed = 0
# for i in range(5):
#     seed += 1
#     features_x = np.load('all_x.npy')
#     labels = np.load('label.npy')
#     features_x,labels = features_x[:],labels[:]
#
#     visualize_with_tsne(features_x, labels, save_path=f'./img3/seed{seed}_1.svg',seed=seed)
#
#     features_lf = np.load('all_lf.npy')
#     labels = np.load('label.npy')
#     features_lf,labels = features_lf[:],labels[:]
#
#     visualize_with_tsne(features_lf, labels, save_path=f'./img3/seed{seed}_2.svg',seed=seed)

seed = 4
features_x = np.load('all_x.npy')
labels = np.load('label.npy')
features_x,labels = features_x[:],labels[:]

visualize_with_tsne(features_x, labels, save_path=f'./images/representation_1.svg',seed=seed)

features_lf = np.load('all_lf.npy')
labels = np.load('label.npy')
features_lf,labels = features_lf[:],labels[:]

visualize_with_tsne(features_lf, labels, save_path=f'./images/representation_2.svg',seed=seed)