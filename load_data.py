import logging
import random
from torchvision import datasets, transforms
import utils.dists as dists


class Generator(object):
    """Generate federated learning training and testing data."""

    # Abstract read function
    def read(self, path):
        # Read the dataset, set: trainset, testset, labels
        raise NotImplementedError

    # Group the data by label
    def group(self):
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}  # pylint: disable=no-member

        # Populate grouped data dict
        for datapoint in self.trainset:  # pylint: disable=all
            _, label = datapoint  # Extract label
            label = self.labels[label]

            grouped_data[label].append(  # pylint: disable=no-member
                datapoint)

        self.trainset = grouped_data  # Overwrite trainset with grouped data

    # Run data generation
    def generate(self, path):
        self.read(path)
        self.trainset_size = len(self.trainset)  # Extract trainset size
        self.group()

        return self.trainset


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator):
        # Get data from generator
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        # print('this label: ', label)
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')
            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []
            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size, type):
        # Get an partition uniform across all labels

        # Use uniform distribution
        if type == 'uniform':
            dist = dists.uniform(partition_size, len(self.labels))
        elif type == 'normal':
            dist = dists.normal(partition_size, len(self.labels))

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        # Return the entire testset
        return self.testset


'''class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref):
        # Get a non-uniform partition with a preference bias

        # Extract bias configuration from config
        bias = self.config.data.bias['primary']
        secondary = self.config.data.bias['secondary']

       # Calculate sizes of majorty and minority portions
        majority = int(partition_size * bias)
        minority = partition_size - majority

        # Calculate number of minor labels
        len_minor_labels = len(self.labels) - 1

        if secondary:
                # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            dist = dists.uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        # dist: [480, 14, 13, 13, 13, 13, 14, 14, 13, 13]
        print('labels: ',self.labels)
        partition = []  # 根据分布提取数据
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition'''


class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, type, pref, bias):

        labels = self.labels

        # Get a non-uniform partition with a preference bias

        # Extract bias configuration from config
        # bias = self.config.data.bias['primary']
        secondary = self.config.data.bias['secondary']

        # Calculate sizes of majorty and minority portions
        majority = int(partition_size * bias)
        minority = partition_size - majority

        # Calculate number of minor labels
        len_minor_labels = len(labels) - 1

        if secondary:
            # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            # dist = dists.uniform(minority, len_minor_labels)
            if type == 'uniform':
                dist = dists.uniform(minority, len_minor_labels)
            elif type == 'normal':
                dist = dists.normal(minority, len_minor_labels)

        # Add majority data to distribution
        # dist.insert(self.labels.index(pref), majority)

        partition = []  # 根据分布提取数据

        if pref in labels:
            dist.insert(labels.index(pref), majority)
            # print('pref: ', pref, '; dist:', dist)
            for i, label in enumerate(labels):
                partition.extend(self.extract(label, dist[i]))
            random.shuffle(partition)

        return partition


class ShardLoader(Loader):
    """Load and pass 'shard' data partitions."""

    def create_shards(self):
        # Extract shard configuration from config
        per_client = self.config.data.shard['per_client']

        # Determine correct total shards, shard size
        total = self.config.clients.total * per_client
        shard_size = int(self.trainset_size / total)

        data = []  # Flatten data
        for _, items in self.trainset.items():
            data.extend(items)

        shards = [data[(i * shard_size):((i + 1) * shard_size)]
                  for i in range(total)]
        random.shuffle(shards)

        self.shards = shards
        self.used = []

        logging.info('Created {} shards of size {}'.format(
            len(shards), shard_size))

    def extract_shard(self):
        shard = self.shards[0]
        self.used.append(shard)
        del self.shards[0]
        return shard

    def get_partition(self):
        # Get a partition shard

        # Extract number of shards per client
        per_client = self.config.data.shard['per_client']

        # Create data partition
        partition = []
        for i in range(per_client):
            partition.extend(self.extract_shard())

        # Shuffle data partition
        random.shuffle(partition)

        return partition
