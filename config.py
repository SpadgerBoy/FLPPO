from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        fields = ['total', 'per_round', 'label_distribution', 'do_test', 'test_partition']
        defaults = (0, 0, 'uniform', False, None)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

        # -- Data --
        fields = ['loading', 'partition', 'IID', 'bias', 'shard']
        defaults = ('static', 0, True, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)


        # Determine correct data loader
        # assert self.data.IID ^ bool(self.data.bias) ^ bool(self.data.shard)
        if self.data.IID:
            self.loader = 'basic'
        elif self.data.bias:
            self.loader = 'bias'
        elif self.data.shard:
            self.loader = 'shard'


        # -- Federated learning --
        fields = ['rounds', 'clients_task', 'target_accuracy', 'epochs', 'batch_size']
        defaults = (10, 'train', 0.99, 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)


        # -- Model --
        self.model = config['model']


        # -- Paths --
        fields = ['data', 'model', 'save_reports']
        defaults = ('./data', './models', False)
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]

        # Set specific model path
        params[fields.index('model')] += '/' + self.model
        self.paths = namedtuple('paths', fields)(*params)

        # -- Server --
        self.server = config['server']

        # -- DQN or PPO--
        if self.server == 'dqn' or self.server == 'ppo':
            pass








