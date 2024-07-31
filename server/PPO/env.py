import os
import numpy as np
import random, time
from threading import Thread
from sklearn.decomposition import PCA
import pickle as pk
from server import Server
from utils.kmeans import kmeans_groups
from utils.network import get_network_delay, delay_to_prob, normalization


class Env(Server):

    def __init__(self, config):
        super().__init__(config)
        self.n_clusters = None
        self.steps = None
        self.groups = []
        self.reward_func = None

        self.output_path = ''
        self.pca_path = ''
        self.global_model_path = ''

        self.pca1_n = self.N
        self.pca1 = PCA(n_components=self.pca1_n)
        self.pca2 = None
        self.pca2_n = None
        self.clients_weights_pca = None

        # dynamical probability
        self.clients_prob = []
        self.clients_prob_weights = []
        self.theta = 0

    def pca1_pca2(self, train_pca, pca1_weights, pca2_groups):

        if train_pca:
            self.pca2 = PCA(n_components=self.pca2_n)
            temp_weights = self.pca2.fit_transform(pca1_weights)
        else:
            temp_weights = self.pca2.transform(pca1_weights)

        new_clients_group_weights = np.zeros((len(pca2_groups), self.pca2_n))

        for i, group in enumerate(pca2_groups):
            new_weights_group = np.zeros((1, self.pca2_n))

            for j, client_id in enumerate(group):
                new_weights_group += temp_weights[client_id]

            new_weights_group = new_weights_group / len(group)
            new_clients_group_weights[i] = new_weights_group

        # print(new_clients_group_weights)
        return new_clients_group_weights

    # 分析每个client的模型权重，并通过pca进行第一次降维，并对其进行
    def profiling(self, clients, train_pca=False):

        # Configure clients to train on local data
        self.configuration(clients, self.global_model_path)

        # Train on local data for profiling purposes
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports, _ = self.reporting(clients)
        # print('clients delay:', self.clients_delay)

        clients_weights = [self.flatten_weights(report.weights) for report in reports]  # list of numpy arrays
        clients_weights = np.array(clients_weights)  # convert to numpy array
        clients_prefs = [report.pref for report in reports]  # 每个客户端中的主导类，dominant class in each client

        # 进行一次pca降维
        if train_pca:  # 在训练期间首次初始化PCA模型，first time to initialize the PCA model during training
            # build the PCA transformer
            print("Start building the PCA transformer...")
            t_start = time.time()
            clients_weights_pca1 = self.pca1.fit_transform(clients_weights)  # 用clients_weights来训练PCA模型，同时返回降维后的数据
            t_end = time.time()
            print("Built PCA transformer, time: {:.2f} s".format(t_end - t_start))

        else:
            clients_weights_pca1 = self.pca1.transform(clients_weights)

        # 分组；pca1前、pca1后都可以
        self.n_clusters = int(self.N / len(self.loader.labels))
        kmeans = kmeans_groups(clients_weights, self.n_clusters, self.K)
        self.groups = kmeans.kmeans_clusters()

        # 进行二次pca降维
        self.pca2_n = len(self.groups)
        clients_weights_pca = self.pca1_pca2(train_pca, clients_weights_pca1, self.groups)

        # 存储pca model
        path = self.pca_path + '/pca1_model.pkl'
        pk.dump(self.pca1, open(path, "wb"))
        path = self.pca_path + '/pca2_model.pkl'
        pk.dump(self.pca2, open(path, "wb"))
        print("PCA model path： ", self.pca_path)

        # 存储通过pca降维后的数据
        path = self.pca_path + '/clients_weights_pca1.pkl'
        pk.dump(clients_weights_pca1, open(path, "wb"))
        path = self.pca_path + '/clients_weights_pca2.pkl'
        pk.dump(clients_weights_pca, open(path, "wb"))

        # 存储每个客户端中的主导类
        path = self.pca_path + '/clients_prefs.pkl'
        pk.dump(clients_prefs, open(path, "wb"))

        # 存储clients分组数据
        path = self.pca_path + '/clients_groups.pkl'
        pk.dump(self.groups, open(path, "wb"))

        # 根据客户端的报告获取服务器模型更新的权重
        server_weights = [Server.flatten_weights(self.aggregation(reports))]
        server_weights = np.array(server_weights)
        server_weights_pca1 = self.pca1.transform(server_weights)
        server_weights_pca = self.pca2.transform(server_weights_pca1)

        self.fileLog.info(f"Total clients: {self.N};   Total groups: {len(self.groups)}")
        self.fileLog.info(f'The grouping of clients is as follows:\n {np.array(self.groups)}')
        self.fileLog.info(f"prefs of clients: {clients_prefs}")
        self.fileLog.info(f"shape of initial clients weights: {clients_weights.shape}")  # (N, 431080)
        self.fileLog.info(f"shape of clients weights after PCA1 dimensionality: {clients_weights_pca1.shape}")
        self.fileLog.info(
            f"shape of clients weights after PCA2 dimensionality: {clients_weights_pca.shape}\n{clients_weights_pca}")
        # self.logger.info(f"shape of server weights after PCA2 dimensionality: {server_weights_pca.shape}")

        return clients_weights_pca, server_weights_pca

    # 获取all clients的pca weights，并得到server model
    def profile_all_clients(self, train_pca):

        print("Start profiling all clients...")

        assert len(self.clients) == self.N

        # 对所有客户端分析其模型
        self.clients_weights_pca, self.server_weights_pca = self.profiling(self.clients, train_pca)

        # 保存每个客户端+服务器的初始pca权重
        self.pca_weights_clientserver_init = np.vstack((self.clients_weights_pca, self.server_weights_pca))

        # self.logger.info(f"shape of client+server weights: {self.pca_weights_clientserver_init.shape}")

        # 保存副本以供以后在训练集中更新
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

    def calculate_reward(self, flag, acc_this_round, old_acc):

        if flag == 1:
            reward = 10 / (0.99 - acc_this_round)
            return reward

        elif flag == 2:
            acc_last_round = old_acc
            if acc_this_round >= acc_last_round:
                reward = 1 / (1 - acc_this_round)
            else:
                reward = (acc_this_round - acc_last_round) / (1 - acc_this_round)
                # reward = 10*(acc_this_round - acc_last_round)/(1 - acc_this_round)
            return reward

        elif flag == 3:
            acc_last_round = old_acc
            if acc_this_round >= acc_last_round:
                reward = 1 / (1 - acc_this_round)
            else:
                v = acc_last_round - acc_this_round
                reward = 1 / ((1 - acc_this_round) * (1000 ** v))
            return reward

        elif flag == 4:
            base_acc = old_acc
            delta = acc_this_round - base_acc
            reward = (100 ** delta - 1) * 50
            return reward

        else:
            raise NameError

    # 进行单个step的训练
    def one_round(self, clients_ids):
        import fl_model

        sample_clients = []
        for i in clients_ids:
            sample_clients.append(self.clients[i])

        # Configure sample clients
        self.configuration(sample_clients, self.global_model_path)

        # Run clients using multithreading for better parallelism
        print("\nTraining on clients: ", clients_ids)
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Receive client updates
        reports, _ = self.reporting(sample_clients)  # list of weight tensors

        # Perform weight aggregation
        self.streamLog.info('Aggregating updates...')
        updated_weights = self.aggregation(reports)
        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)
        # save the updated global model for use in the next communication round
        self.save_model(self.model, self.global_model_path)

        print('Test updated model on server...')
        testset = self.loader.get_testset()
        batch_size = self.config.fl.batch_size
        testloader = fl_model.get_testloader(testset, batch_size)
        accuracy = fl_model.test(self.model, testloader)
        # self.streamLog.info('Average accuracy: {:.2f}%'.format(100 * accuracy))

        # 获取next_state用于agent的网络输入
        next_state = self.clients_weights_pca[:, 0]
        # next_state = self.pca_weights_clientserver.flatten()
        # print("next_state.shape:", next_state.shape)
        next_state = next_state.tolist()
        next_state.append(accuracy)

        return accuracy, next_state

    def step(self, clients, old_acc):

        accuracy, next_state = self.one_round(clients)

        reward = self.calculate_reward(self.reward_func, accuracy, old_acc)

        return next_state, accuracy, reward

    # 每个Episode开始前先初始化状态
    def reset_state(self):

        next_state = self.clients_weights_pca[:, 0]
        next_state = next_state.tolist()
        next_state.append(0)

        return next_state

    def action_to_clients(self, action, train=True):

        if train:
            clients = self.groups[action]
        else:
            if len(self.groups[action]) <= self.K:
                clients = self.groups[action]
            else:
                clients = np.random.choice(self.groups[action], size=self.K, replace=False,
                                           p=self.clients_prob[action])
                # 调整概率
            self.dynamic_tuning(action, clients)

        return clients

    def dynamic_tuning(self, action, clients_list):

        for j, client_id in enumerate(clients_list):
            for i, client in enumerate(self.groups[action]):
                if client_id == client:
                    self.clients_prob_weights[action][i] *= self.theta

        old_weights = self.clients_prob_weights[action].copy()
        new_prob = normalization(old_weights)
        self.clients_prob[action] = new_prob
