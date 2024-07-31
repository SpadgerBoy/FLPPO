
class GeneralConfig:
    def __init__(self) -> None:
        self.wrapper = None     # wrapper of environment
        # self.render = False     # whether to render environment
        self.device = "cuda"    # device to use

        self.mode = "Train"  # Train or Test
        # self.mode = "Test"
        self.load_checkpoint = False         # True or False
        # self.load_checkpoint = True
        self.load_path = './output/100/Train-40-10'

        # train
        self.max_steps = 1    # max steps for each episode
        self.train_eps = 1    # number of episodes for training
        self.eval_per_episode = 10  # evaluation per episode
        self.eval_eps = 5  # number of episodes for evaluation
        self.reward_func = 2  # reward-functions: 1, 2, 3

        # test
        self.test_rounds = 10  # number of episodes for testing

        # consider dynamic network connections and network delay
        self.dynamic_network = False

        # Discount rate for selecting client probability
        self.theta = 0.4


class AlgoConfig:
    def __init__(self):
        self.ppo_type = 'clip'  # clip
        self.gamma = 0.97   # discount factor

        self.k_epochs = 5   # update policy for K epochs
        self.actor_lr = 0.003   # learning rate for actor
        self.critic_lr = 0.01  # learning rate for critic

        self.eps_clip = 0.3     # clip parameter for PPO
        self.entropy_coef = 0.03    # entropy coefficient

        self.train_batch_size = 150     # ppo train batch size,经验回放池大小
        self.sgd_batch_size = 64    # sgd batch size

        self.actor_hidden_dim = 128     # hidden dimension for actor
        self.critic_hidden_dim = 128    # hidden dimension for critic
