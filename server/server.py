import client
import load_data
import numpy as np
import pickle
import random, time
import sys
from threading import Thread
import torch

import utils.dists as dists  # pylint: disable=no-name-in-module
from utils.MyThread import ThreadWithReturnValue
from utils.plot import plot_acc
from utils.log import mylogger
from utils.network import get_network_delay, delay_to_prob, normalization


class Server(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config
        self.N = self.config.clients.total
        self.K = self.config.clients.per_round

        self.if_save_report = self.config.paths.save_reports
        self.clients_sleeping = []
        self.groups = []
        self.fileLog = None
        self.streamLog = None
        self.clients_delay = []

    # Set up server
    def boot(self, cfg):

        self.fileLog = mylogger(1, cfg.log_dir)
        self.streamLog = mylogger(2, '')

        self.fileLog.info(f'Booting {cfg.method} server...')

        fl_model_path = self.config.paths.model
        total_clients = self.N

        # Add fl_model to import path: import fl_model
        sys.path.append(fl_model_path)

        # Set up simulated server
        self.load_data()
        self.fileLog.info(f'Model: {self.config.model}')
        self.load_model(cfg.global_model_dir)  # save initial global model
        self.make_clients(total_clients)

    def load_data(self):
        import fl_model

        # Extract config for loaders
        config = self.config

        # Set up data generator
        generator = fl_model.Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels

        self.fileLog.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        self.fileLog.info('Labels ({}): {}'.format(len(labels), labels))

        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),  # IID
            'bias': load_data.BiasLoader(config, generator),  # NON-IID
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]

        self.fileLog.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self, global_model_path):
        import fl_model  # pylint: disable=import-error

        # model_path = self.config.paths.model
        model_path = global_model_path

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path)

        # Extract flattened weights (if applicable)
        if self.if_save_report:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading
        labels1 = labels

        # 处理数据分布
        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels1)),
                "normal": dists.normal(num_clients, len(labels1))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

            # dist是选中每个label的概率分布

        # Make simulated clients
        clients = []
        for client_id in range(num_clients):

            # new_client = client.Client(client_id, self.case_name)
            new_client = client.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    # pref = random.choices(labels1, dist)[0]  # 按照dist中的概率从labels中随机选择一个主成分
                    k = client_id % 10
                    pref = labels[k]
                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)

                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard
                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        self.fileLog.info(f'Total clients: {len(clients)}')
        self.fileLog.info(f'Data distribution in each client: {self.config.clients.label_distribution}')

        if loader == 'bias':
            self.fileLog.info('Pref distribution across all clients: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()
            # Send data partition to all clients
            # [self.set_client_data(client) for client in clients]
            if not IID:
                bias_list = self.config.data.bias['primary']
                for c in clients:
                    flag = int(c.client_id / 10)
                    self.set_client_data(c, bias_list[flag])
            else:
                for c in clients:
                    self.set_client_data(c, 0)

        self.clients = clients

    # Run federated learning
    def run(self, cfg):

        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        self.clients_delay = get_network_delay(self.N, [0.3, 0.5, 0.2])

        if target_accuracy:
            self.fileLog.info(f'Training: {rounds} rounds or {100 * target_accuracy}% accuracy\n')

        with open(cfg.res_dir + '/res.csv', 'w') as f:
            f.write('time(s),Rounds,Accuracy\n')

        # Perform rounds of federated learning
        time_consuming = 0.0
        for round in range(1, rounds + 1):
            print(f'\n**** Round {round}/{rounds} ****')

            accuracy, time_delays = self.round(cfg)

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
                pickle.dump(self.saved_reports, f)
            print(f'\nSaved all reports: {reports_path}')

        print(f"\nFinish {cfg.method} Testing in {self.config.model} dataset!")

    def round(self, cfg):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_clients = self.selection()
        sample_clients_ids = [client.client_id for client in sample_clients]
        print("Training on clients: ", sample_clients_ids)

        time_delays = max([self.clients_delay[client_id] for client_id in sample_clients_ids])

        # Configure sample clients
        self.configuration(sample_clients, cfg.global_model_dir)

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
        self.save_model(self.model, cfg.global_model_dir)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average test accuracy from client reports
            # print('Get average accuracy from client reports')
            accuracy = self.accuracy_averaging(reports)

        else:  # Test updated model on server using the aggregated weights
            # print('Test updated model on server')
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        # self.streamLog.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy, time_delays

    # Federated learning phases
    def selection(self):

        # Select clients randomly
        sample_clients = [client for client in random.sample(self.clients, self.K)]

        return sample_clients

    def configuration(self, sample_clients, global_model_path):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                pass
                # self.set_client_data(client, )  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config, global_model_path)  # load global model to sampled client

    def reporting(self, sample_clients):
        reports = [client.get_report() for client in sample_clients]
        self.streamLog.info('Reports received: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)
        return reports, 0

    '''
    def reporting(self, sample_clients):
        threads = [None] * len(sample_clients)
        reports = []
        spend_times = [None] * self.N  # 用于记录每个client的延迟
        for i in range(len(sample_clients)):
            client = sample_clients[i]
            sleeping = self.clients_sleeping[client.client_id]
            threads[i] = ThreadWithReturnValue(client.get_report, args=(sleeping,))
            threads[i].start()
        for i in range(len(sample_clients)):
            client = sample_clients[i]
            threads[i].join()
            report, spend_time = threads[i].get_result()
            reports.append(report)
            spend_times[client.client_id] = spend_time

        # reports = [client.get_report() for client in sample_clients]
        self.streamLog.info('Reports received: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)
        return reports, spend_times
    '''

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_client_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size()) for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client, bias):
        loader = self.config.loader
        type = self.config.clients.label_distribution

        # Get data partition size
        partition_size = ''
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for client
        if loader == 'basic':
            data = self.loader.get_partition(partition_size, type)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, type, client.pref, bias)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            self.streamLog.critical('Unknown data loader type')

        # Send data to client
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/global_model.h5'
        torch.save(model.state_dict(), path)
        self.streamLog.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))

