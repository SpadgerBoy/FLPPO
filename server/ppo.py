import copy
import random,time
from pathlib import Path
import pickle as pk

from server.PPO.agent import Agent
from server.PPO.env import Env
from utils.plot import plot_rewards, plot_acc, plot_pca, plot_pca_groups
from utils.copyfile import copyFiles


class PPOServer(Env):
    def __init__(self, config):
        super().__init__(config)
        self.clients_prob_init = []
        self.clients_prob_weights_init = []

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
            clients = self.action_to_clients(action, train=True)
            print(f'states: {state},    action:{action}')

            next_state, acc, reward = self.step(clients, last_acc)
            total_reward += reward
            com_rounds += 1
            last_acc = acc
            # total_delay += delay

            if acc >= cfg.target_acc or t == cfg.max_steps-1:
                done = True

            #self.base_acc[episode_ct-1].append(acc)
            print(f"Train: [Episode:{i_eps}/{cfg.train_eps},  step:{t + 1}/{cfg.max_steps},  acc:{acc:.4f},  "
                  f"total reward:{total_reward:.2f},  clients:{clients}]\n")

            agent.memory.push((state, action, reward, done, agent.probs, agent.log_probs))

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
            clients = self.action_to_clients(action, train=True)
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
            f.write('Rounds,Accuracy\n')

        actions = []

        for r in range(1, cfg.test_rounds+1):

            action = agent.predict_action(state)  # sample action
            actions.append(action)
            clients = self.action_to_clients(action, train=False)

            next_state, acc, reward = self.step(clients, last_acc)

            last_acc = acc
            state = next_state  # update next state for env

            with open(fn, 'a') as f:
                f.write(f'{r},{acc:.4f}\n')
            self.fileLog.info(f"Round: {r}/{cfg.test_rounds}, Acc: {acc: .4f}, clients:{clients}")

        return agent

    # 计算预测的平均奖励
    def evaluate(self, cfg, agent):
        sum_eval_reward = 0
        sum_eval_acc = 0
        for i in range(1, cfg.eval_eps+1):
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
        best_model_dir = f"{cfg.model_dir}/best_models"
        setattr(cfg, 'best_model_dir', best_model_dir)
        pca_dir = f"{cfg.output_dir}/pca"
        setattr(cfg, 'pca_dir', pca_dir)


        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
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
            plot_pca(cfg.pca_dir, self.N, self.config.model)
            plot_pca_groups(cfg.pca_dir, len(self.groups), self.N)

        # 获取每组中客户端被选择的概率
        clients_prob_weights = []
        clients_prob = []
        for group in self.groups:
            clients_prob_weights.append([1] * len(group))
            prob = 1 / len(group)
            clients_prob.append([prob] * len(group))
        self.clients_prob_init = clients_prob
        self.clients_prob_weights_init = clients_prob_weights

        # 获取搭建agent模型所需的参数
        self.n_states = len(self.groups) + 1
        setattr(cfg, 'n_states', self.n_states)
        self.n_actions = len(self.groups)
        setattr(cfg, 'n_actions', self.n_actions)
        self.fileLog.info(f'states:{cfg.n_states},   actions:{cfg.n_actions}')

        agent = Agent(cfg)

        if cfg.load_checkpoint:
            agent.load_model(f"{cfg.load_path}/ppo_models")

        if cfg.mode.lower() == 'train':

            Path(cfg.best_model_dir).mkdir(parents=True, exist_ok=True)

            print('Episodes: ', cfg.train_eps, ',      Steps: ', cfg.max_steps)
            best_ep_reward = -float('inf')
            fn = f'{cfg.res_dir}/res.csv'
            with open(fn, 'w') as f:
                f.write('Episodes,Reward,Accuracy\n')

            for i_ep in range(1, cfg.train_eps+1):

                self.clients_prob = copy.deepcopy(self.clients_prob_init)
                self.clients_prob_weights = copy.deepcopy(self.clients_prob_weights_init)

                agent, ep_reward, acc, actions = self.train_one_episode(agent, cfg, i_ep)

                with open(fn, 'a') as f:
                    f.write(f'{i_ep},{ep_reward:.2f},{acc:.4f}\n')

                agent.save_model(cfg.model_dir)
                self.fileLog.info(f"Episode: {i_ep}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Accuracy: {acc:.4f}, actions:{actions}")

                # Evaluate
                if i_ep % cfg.eval_per_episode == 0:
                    print('\n------------------------Eval start------------------------')
                    mean_eval_reward, mean_eval_acc = self.evaluate(cfg, agent)
                    self.fileLog.info(f"Eval by {cfg.eval_eps} episodes: mean of eval reward: {mean_eval_reward:.3f}, mean of eval accuracy: {mean_eval_acc:.4f}")
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        self.fileLog.info(f"Current episode {i_ep} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(cfg.best_model_dir)  # save models with best reward
                    print('------------------------Eval  end------------------------')

            plot_rewards(cfg.mode.lower(), fpath=cfg.res_dir)
            plot_acc(cfg.mode.lower(), fpath=cfg.res_dir)

        elif cfg.mode.lower() == 'test':

            print(f'Total Test Rounds: {cfg.test_rounds}\n\n')

            self.clients_prob = copy.deepcopy(self.clients_prob_init)
            self.clients_prob_weights = copy.deepcopy(self.clients_prob_weights_init)

            agent = self.test_rounds(agent, cfg)
            agent.save_model(cfg.model_dir)

            plot_acc(cfg.mode.lower(), fpath=cfg.res_dir)

        print(f"\nFinish {cfg.method} {cfg.mode}ing in {self.config.model} dataset!")
