import os
import numpy as np
import random, time
from threading import Thread
from sklearn.decomposition import PCA
import pickle as pk
from pathlib import Path

from server import Server
from utils.kmeans import kmeans_groups
from utils.plot import plot_acc, plot_pca
from utils.copyfile import copyFiles
from utils.network import get_network_delay, delay_to_prob, normalization


class Cluster(Server):
    def __init__(self, config):
        super().__init__(config)
        self.n_clusters = None
        self.steps = None
        self.groups = []

        self.pca_path = ''
        self.global_model_path = ''
        self.pca1_n = self.N
        self.pca1 = PCA(n_components=self.pca1_n)  # 返回所保留的成分个数n
        self.clients_weights_pca = None

        # 分析每个client的模型权重，并通过pca进行第一次降维，并对其进行

    def profile_all_clients(self, train_pca=False):

        assert len(self.clients) == self.N

        # Configure clients to train on local data
        self.configuration(self.clients, self.global_model_path)

        # Train on local data for profiling purposes
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports, _ = self.reporting(self.clients)
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

        # 存储pca model
        path = self.pca_path + '/pca1_model.pkl'
        pk.dump(self.pca1, open(path, "wb"))
        print("PCA model path： ", self.pca_path)

        # 存储通过pca降维后的数据
        path = self.pca_path + '/clients_weights_pca1.pkl'
        pk.dump(clients_weights_pca1, open(path, "wb"))

        # 存储每个客户端中的主导类
        path = self.pca_path + '/clients_prefs.pkl'
        pk.dump(clients_prefs, open(path, "wb"))

        # 存储clients分组数据
        path = self.pca_path + '/clients_groups.pkl'
        pk.dump(self.groups, open(path, "wb"))

        self.fileLog.info(f"Total clients: {self.N};   Total groups: {len(self.groups)}")
        self.fileLog.info(f'The grouping of clients is as follows:\n {np.array(self.groups)}')
        self.fileLog.info(f"prefs of clients: {clients_prefs}")
        self.fileLog.info(f"shape of initial clients weights: {clients_weights.shape}")  # (N, 431080)
        self.fileLog.info(f"shape of clients weights after PCA dimensionality: {clients_weights_pca1.shape}")

    def action_to_clients(self, action):
        if len(self.groups[action]) <= self.K:
            clients = self.groups[action]
        else:
            clients = np.random.choice(self.groups[action], size=self.K, replace=False)
        return clients

    def round(self, action):

        import fl_model  # pylint: disable=import-error

        clients_id = self.action_to_clients(action)
        sample_clients = []
        for i in clients_id:
            sample_clients.append(self.clients[i])
        print("Training on clients: ", clients_id)

        # Configure sample clients
        self.configuration(sample_clients, self.global_model_path)

        time_delays = max([self.clients_delay[client_id] for client_id in clients_id])

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Receive client updates
        reports, _ = self.reporting(sample_clients)

        if not self.config.data.IID:
            print("clients' prefs: ", [report.pref for report in reports])

        # Perform weight aggregation
        self.streamLog.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.if_save_report:
            self.save_reports(round, reports)

        # Save updated global modelc
        self.save_model(self.model, self.global_model_path)

        # Test global model accuracy

        print('Test updated model on server')
        testset = self.loader.get_testset()
        batch_size = self.config.fl.batch_size
        testloader = fl_model.get_testloader(testset, batch_size)
        accuracy = fl_model.test(self.model, testloader)

        # self.streamLog.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy, time_delays

    # 生成输出文件夹
    def create_dirs(self, cfg):

        pca_dir = f"{cfg.output_dir}/pca"
        setattr(cfg, 'pca_dir', pca_dir)

        Path(cfg.pca_dir).mkdir(parents=True, exist_ok=True)

        self.pca_path = pca_dir
        self.global_model_path = cfg.global_model_dir

    def run(self, cfg):

        self.create_dirs(cfg)

        setattr(cfg, 'total_clients', self.N)

        # 初始化所有clients
        self.profile_all_clients(train_pca=True)
        plot_pca(cfg.pca_dir, self.N, self.config.model)

        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        self.clients_delay = get_network_delay(self.N, [0.3, 0.5, 0.2])

        if target_accuracy:
            self.fileLog.info(f'Training: {rounds} rounds or {100 * target_accuracy}% accuracy\n')

        with open(cfg.res_dir + '/res.csv', 'w') as f:
            f.write('time(s),Rounds,Accuracy\n')

        time_consuming = 0.0
        for round in range(1, rounds + 1):
            print(f'\n**** Round {round}/{rounds} ****')

            action = np.random.choice(len(self.groups))

            accuracy, time_delays = self.round(action)

            self.fileLog.info(f"Round: {round}/{rounds}, Accuracy: {accuracy:.4f}")

            time_consuming = time_consuming + time_delays

            with open(cfg.res_dir + '/res.csv', 'a') as f:
                f.write(f'{time_consuming:.2f},{round},{accuracy:.4f}\n')

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                self.fileLog.info('Target accuracy reached.')
                break

        plot_acc('test', fpath=cfg.res_dir)

        if self.if_save_report:
            from pathlib import Path
            Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
            reports_path = cfg.reports_dir + '/weights.pkl'
            with open(reports_path, 'wb') as f:
                pk.dump(self.saved_reports, f)
            print(f'\nSaved all reports: {reports_path}')

        print(f"\nFinish {cfg.method} testing!")
