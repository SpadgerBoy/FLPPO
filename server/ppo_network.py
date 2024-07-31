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

        # self.n_clusters = int(self.N/self.K)
        self.fileLog = ''  # create the logger
        self.n_clusters = None
        self.steps = None
        self.groups = []
        self.reward_func = None
        self.output_path = ''
        self.pca_path = ''
        self.global_model_path = ''
        self.pca1_n = self.N
        self.pca1 = PCA(n_components=self.pca1_n)  # 返回所保留的成分个数n
        self.pca2 = None
        self.pca2_n = None
        self.clients_weights_pca = None

        self.clients_delay = []
        self.clients_network_weights = []
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
        # 仅更新self.pca_weights_clientserver_init中所选设备的权重
        # self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()
        # next_state = self.pca_weights_clientserver.flatten()
        next_state = self.clients_weights_pca[:, 0]
        next_state = next_state.tolist()
        next_state.append(0)

        return next_state

    def action_to_clients(self, action):
        if len(self.groups[action]) <= self.K:
            clients = self.groups[action]
        else:
            # clients = np.random.choice(self.groups[action], size=self.K, replace=False)
            clients = np.random.choice(self.groups[action], size=self.K, replace=False,
                                       p=self.clients_network_weights[action])

        self.dynamic_tuning(action, clients)
        return clients

    def dynamic_tuning(self, action, clients_list):
        for j, client_id in enumerate(clients_list):
            for i, client in enumerate(self.groups[action]):
                if client_id == client:
                    self.clients_network_weights[action][i] *= self.theta
        g_network_weights = normalization(self.clients_network_weights[action])
        self.clients_network_weights[action] = g_network_weights


import random, time
from pathlib import Path
# import pickle as pk

# from server.RL.env import Env
from server.PPO.agent import Agent
from utils.plot import plot_rewards, plot_acc, plot_pca, plot_pca_groups
from utils.copyfile import copyFiles
from utils.network import get_network_delay, delay_to_prob, normalization


class PPOnServer(Env):
    def __init__(self, config):
        super().__init__(config)
        self.clients_network_weights_init = None
        self.clients_delay_init = None

    # Train
    def train_one_episode(self, agent, cfg, i_eps):

        print(f'\n\n****************************Train: Episode: {i_eps}/{cfg.train_eps}****************************')

        self.load_model(cfg.global_model_dir)  # save initial global model
        state = self.reset_state()

        last_acc = 0
        total_reward = 0
        com_rounds = 0
        acc = 0
        total_delay = 0
        done = False
        actions = []

        for t in range(cfg.max_steps):
            action = agent.sample_action(state)
            actions.append(action)
            clients = self.action_to_clients(action)
            print(f'states: {state},    action:{action}')

            next_state, acc, reward = self.step(clients, last_acc)
            total_reward += reward
            com_rounds += 1
            last_acc = acc
            # total_delay += delay

            if acc >= cfg.target_acc or t == cfg.max_steps - 1:
                done = True

            print(f"Train: [Episode:{i_eps}/{cfg.train_eps},  step:{t + 1}/{cfg.max_steps},  acc:{acc:.4f},  "
                  f"total reward:{total_reward:.2f},  clients:{clients}]\n")

            agent.memory.push((state, action, reward, done, agent.probs, agent.log_probs))
            # print(agent.probs, agent.log_probs)
            agent.update()  # sample a mini-batch from the replay buffer to train the DQN model
            # print(agent.memory.__len__())

            state = next_state

        return agent, total_reward, acc, actions

    # Evaluate
    def eval_one_episode(self, agent, cfg, i_eps):

        print(f"\n\n****************Eval: Episode: {i_eps}/{cfg.eval_eps}****************")

        self.load_model(cfg.global_model_dir)  # save initial global model
        state = self.reset_state()

        last_acc = 0
        ep_reward = 0  # reward per episode
        acc = 0

        for t in range(cfg.max_steps):
            action = agent.predict_action(state)  # sample action
            clients = self.action_to_clients(action)
            print(f'states: {state},    action:{action}')

            next_state, acc, reward = self.step(clients, last_acc)
            state = next_state  # update next state for env
            ep_reward += reward  #
            last_acc = acc
            print(f"Eval: [episode:{i_eps}/{cfg.eval_eps},  step:{t + 1}/{cfg.max_steps},  acc:{acc:.4f},  "
                  f"total reward:{ep_reward:.2f},  clients:{clients}]\n")

        return agent, ep_reward, acc

    # Test
    def test_rounds(self, agent, cfg):

        self.load_model(cfg.global_model_dir)  # save initial global model
        state = self.reset_state()
        last_acc = 0

        fn = f'{cfg.res_dir}/res.csv'
        with open(fn, 'w') as f:
            f.write('time(s),Rounds,Accuracy\n')

        actions = []
        time_consuming = 0

        for r in range(1, cfg.test_rounds + 1):

            action = agent.predict_action(state)  # sample action
            actions.append(action)
            clients = self.action_to_clients(action)

            time_delays = max([self.clients_delay[client_id] for client_id in clients])
            time_consuming = time_consuming + time_delays

            next_state, acc, reward = self.step(clients, last_acc)

            last_acc = acc
            state = next_state  # update next state for env

            with open(fn, 'a') as f:
                f.write(f'{time_consuming:.2f},{r},{acc:.4f}\n')

            # print("rounds:", round + 1,", action:", action, ", acc:", acc, ", total reward:",total_reward)
            self.fileLog.info(f"Round: {r}/{cfg.test_rounds}, Acc: {acc: .4f}, clients:{clients}")
        return agent

    # 计算预测的平均奖励
    def evaluate(self, cfg, agent):
        sum_eval_reward = 0
        sum_eval_acc = 0
        for i in range(1, cfg.eval_eps + 1):
            _, eval_ep_reward, acc = self.eval_one_episode(agent, cfg, i)
            sum_eval_reward += eval_ep_reward
            sum_eval_acc += acc
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        mean_eval_acc = sum_eval_acc / cfg.eval_eps
        return mean_eval_reward, mean_eval_acc

    # 生成输出文件夹
    def create_dirs(self, cfg):
        model_dir = f"{cfg.output_dir}/ppo_models"
        setattr(cfg, 'model_dir', model_dir)
        actor_model_dir = f"{cfg.model_dir}/actor_models"
        setattr(cfg, 'actor_model_dir', actor_model_dir)
        pca_dir = f"{cfg.output_dir}/pca"
        setattr(cfg, 'pca_dir', pca_dir)

        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.actor_model_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.pca_dir).mkdir(parents=True, exist_ok=True)

        self.output_path = cfg.output_dir
        self.pca_path = pca_dir
        self.global_model_path = cfg.global_model_dir

    def run(self, cfg):

        self.create_dirs(cfg)

        self.reward_func = cfg.reward_func
        self.theta = cfg.theta

        setattr(cfg, 'target_acc', self.config.fl.target_accuracy)
        setattr(cfg, 'total_clients', self.N)

        if cfg.load_checkpoint or cfg.mode.lower() == 'test':
            self.pca1 = pk.load(open(f"{cfg.load_path}/pca/pca1_model.pkl", 'rb'))
            self.pca2 = pk.load(open(f"{cfg.load_path}/pca/pca2_model.pkl", 'rb'))
            self.clients_weights_pca = pk.load(open(f"{cfg.load_path}/pca/clients_weights_pca2.pkl", 'rb'))
            self.groups = pk.load(open(f"{cfg.load_path}/pca/clients_groups.pkl", 'rb'))
            copyFiles(f"{cfg.load_path}/pca", f"{cfg.pca_dir}")
            # self.profile_all_clients(train_pca=False)
        else:
            self.profile_all_clients(train_pca=True)
            # plot_pca(cfg.pca_dir, self.N, self.config.model)
            # plot_pca_groups(cfg.pca_dir, len(self.groups), self.N)

        # 设置每组中客户端被选择的概率
        # self.clients_delay_init = get_network_delay(self.N, [0.3, 0.5, 0.2])
        # self.clients_network_weights_init = delay_to_weights(self.clients_delay, self.groups)
        self.clients_delay_init = [3.7, 4.5, 3.8, 8.6, 0.9, 9.5, 4.7, 3.0, 2.4, 2.4,
                                   0.5, 3.6, 0.2, 1.0, 9.8, 4.4, 2.8, 0.8, 0.2, 0.5,
                                   3.5, 1.4, 10.2, 3.3, 3.2, 2.3, 0.3, 4.7, 73.8, 8.6,
                                   0.1, 4.7, 0.2, 1.7, 2.8, 43.2, 3.3, 0.9, 8.5, 0.1]
        self.clients_network_weights_init = [
            [0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.06896551724137931, 0.13793103448275862,
             0.06896551724137931, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896, 0.10344827586206896],
            [0.11428571428571428, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.05714285714285714,
             0.08571428571428572, 0.08571428571428572, 0.11428571428571428, 0.11428571428571428, 0.11428571428571428],
            [0.11432926829268292, 0.11432926829268292, 0.07469512195121951, 0.11432926829268292, 0.11432926829268292,
             0.11432926829268292, 0.1524390243902439, 0.11432926829268292, 0.010670731707317074, 0.07621951219512195],
            [0.13131976362442546, 0.0984898227183191, 0.13131976362442546, 0.0984898227183191, 0.0984898227183191,
             0.01510177281680893, 0.0984898227183191, 0.13131976362442546, 0.06565988181221273, 0.13131976362442546]]
        # self.clients_network_weights_init = [[0.1] * 10, [0.1] * 10, [0.1] * 10, [0.1] * 10]

        # 获取搭建agent模型所需的参数
        # self.n_states = (len(self.groups) + 1) * len(self.groups)
        self.n_states = len(self.groups) + 1
        setattr(cfg, 'n_states', self.n_states)
        self.n_actions = len(self.groups)
        setattr(cfg, 'n_actions', self.n_actions)
        self.fileLog.info(f'states:{cfg.n_states},   actions:{cfg.n_actions}')

        agent = Agent(cfg)

        if cfg.load_checkpoint:
            # agent.load_model(f"{cfg.load_path}/ppo_models")
            agent.load_model(f"{cfg.load_path}/ppo_models")

        if cfg.mode.lower() == 'train':

            print('Episodes: ', cfg.train_eps, ',      Steps: ', cfg.max_steps)

            best_ep_reward = -float('inf')

            fn = f'{cfg.res_dir}/res.csv'
            with open(fn, 'w') as f:
                f.write('Episodes,Reward,Accuracy\n')

            for i_ep in range(1, cfg.train_eps + 1):

                self.clients_delay = self.clients_delay_init
                self.clients_network_weights = self.clients_network_weights_init

                agent, ep_reward, acc, actions = self.train_one_episode(agent, cfg, i_ep)

                with open(fn, 'a') as f:
                    f.write(f'{i_ep},{ep_reward:.2f},{acc:.4f}\n')

                agent.save_model(cfg.actor_model_dir)
                self.fileLog.info(
                    f"Episode: {i_ep}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Accuracy: {acc:.4f}, actions:{actions}")

                # Evaluate
                if i_ep % cfg.eval_per_episode == 0:
                    print('\n------------------------Eval start------------------------')
                    mean_eval_reward, mean_eval_acc = self.evaluate(cfg, agent)
                    self.fileLog.info(
                        f"Eval by {cfg.eval_eps} episodes: mean of eval reward: {mean_eval_reward:.3f}, mean of eval accuracy: {mean_eval_acc:.4f}")
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        self.fileLog.info(f"Current episode {i_ep} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(cfg.model_dir)  # save models with best reward
                    print('------------------------Eval  end------------------------')

            plot_rewards(cfg.mode.lower(), fpath=cfg.res_dir)
            plot_acc(cfg.mode.lower(), fpath=cfg.res_dir)

        elif cfg.mode.lower() == 'test':
            self.clients_delay = self.clients_delay_init
            self.clients_network_weights = self.clients_network_weights_init

            print(f'Total Test Rounds: {cfg.test_rounds}\n\n')
            agent = self.test_rounds(agent, cfg)
            agent.save_model(cfg.model_dir)

            plot_acc(cfg.mode.lower(), fpath=cfg.res_dir)

        print(f"\nFinish {cfg.method} {cfg.mode}ing in {self.config.model} dataset!")
