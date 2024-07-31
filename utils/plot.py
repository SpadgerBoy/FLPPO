import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle


def smooth(data, weight=0.95):
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(cfg_mode, title="rewards", fpath=None):
    sns.set()
    f = pd.read_csv(f'{fpath}/res.csv')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    if cfg_mode == 'train':
        plt.xlabel('Episodes')
    if cfg_mode == 'test':
        plt.xlabel('Rounds')
    plt.plot(f.Reward, label='rewards')
    plt.plot(smooth(f.Reward), label='smoothed')

    plt.legend()
    plt.savefig(f"{fpath}/rewards.png")


def plot_acc(cfg_mode, title="accuracy", fpath=None):
    sns.set()
    f = pd.read_csv(f'{fpath}/res.csv')

    plt.figure()
    plt.title(f"{title}")
    if cfg_mode == 'train':
        plt.xlabel('Episodes')
    if cfg_mode == 'test':
        plt.xlabel('Rounds')

    plt.plot(f.Accuracy)

    if cfg_mode == 'train':
        plt.plot(smooth(f.Accuracy))

    plt.legend()
    plt.savefig(f"{fpath}/accuracy.png")


def plot_pca(pca_dir, total_clients, dataset):

    if dataset == 'MNIST':
        dataset_labels = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight','9 - nine']
    elif dataset == 'FashionMNIST':
        dataset_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        NameError

    with open(f"{pca_dir}/clients_weights_pca1.pkl", 'rb') as a:
        data_weights = pickle.load(a)

    with open(f"{pca_dir}/clients_prefs.pkl", 'rb') as b:
        data_labels = pickle.load(b)

    x1 = data_weights[:, 0]
    y1 = data_weights[:, 1]
    labels = np.array(data_labels)

    total1 = np.zeros((2, total_clients))
    total1[0] = x1
    total1[1] = y1

    color = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "cyan", "brown"])
    total1 = np.transpose(total1)

    fig, ax = plt.subplots()
    for i, label in enumerate(dataset_labels):
        pref_label = f"Pref: {label}"
        index = np.where(labels == label)
        for j in index:
            ax.scatter(total1[j, 0], total1[j, 1], c=color[i], label=pref_label)

    ax.legend()
    plt.title(f'PCA on the {dataset} dataset({total_clients}-clients)')
    plt.xlabel('C0')
    plt.ylabel('C1')

    plt.savefig(f"{pca_dir}/{total_clients}-clients.png")

def plot_pca_groups(pca_dir, total_groups, total_clients):

    with open(f"{pca_dir}/clients_weights_pca2.pkl", 'rb') as a:
        data_weight1 = pickle.load(a)

    with open(f"{pca_dir}/clients_groups.pkl", 'rb') as b:
        data_label1 = pickle.load(b)

    x1 = data_weight1[:, 0]
    y1 = data_weight1[:, 1]
    l1 = data_label1
    total1 = np.zeros((2, total_groups))
    total1[0] = x1
    total1[1] = y1
    color = np.array(["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "cyan", "brown"])
    total1 = np.transpose(total1)

    fig, ax = plt.subplots()

    for i, group in enumerate(l1):
        label = "Group: " + str(i)
        ax.scatter(total1[i, 0], total1[i, 1], c=color[i], label=label)

    ax.legend()
    plt.title(f'PCA on the dataset({total_groups}-groups/{total_clients}-clients)')
    plt.xlabel('C0')
    plt.ylabel('C1')

    plt.savefig(f"{pca_dir}/{total_clients}-clients____{total_groups}-groups.png")


def plot_losses(losses, algo="PPO", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('Episode')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()

