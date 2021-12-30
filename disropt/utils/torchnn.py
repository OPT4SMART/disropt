import torch
from torch.utils.data import Sampler, Dataset
from typing import Iterator
from math import ceil
from statistics import mean


class DistributedSampler(Sampler[int]):
    """Class that distributes a dataset among the agents (equivalent to Tensorflow shard).
    Note: the length of the dataset must be divisible by the size of the network.
    """

    def __init__(self, dataset: Dataset, n_agents: int,
                 agent_id: int, n_samples: int = None, shuffle: bool = False,
                 seed: int = 0) -> None:
        len_dataset = len(dataset) if n_samples is None else n_samples
        self.local_size = ceil(len_dataset / n_agents)
        self.shuffle = shuffle

        if shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        
        ## PRE-COMPUTE LIST OF LOCAL INDICES
        total_size = self.local_size * n_agents

        # initialize list of all indices
        self.indices = list(range(len_dataset))
        assert len(self.indices) == total_size

        # subsample
        self.indices = torch.tensor(self.indices[agent_id:total_size:n_agents], dtype=torch.long)
        assert len(self.indices) == self.local_size

    def __iter__(self) -> Iterator[int]:
        indices = self.indices.clone()

        if self.shuffle:
            indices = indices[torch.randperm(len(indices), generator=self.generator)]

        return iter(indices)

    def __len__(self) -> int:
        return self.local_size

class MeanMetric:
    def __init__(self) -> None:
        self.state = []
    
    def reset_states(self) -> None:
        self.state = []
    
    def update_state(self, value) -> None:
        self.state.append(float(value))
    
    def result(self) -> float:
        return mean(self.state)

class CategoricalAccuracyMetric:
    def __init__(self) -> None:
        self.correct = 0.0
        self.num_samples = 0.0
    
    def reset_states(self) -> None:
        self.correct = 0.0
        self.num_samples = 0.0
    
    def update_state(self, labels, predictions) -> None:
        self.correct += (predictions.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
        self.num_samples += labels.shape[0]
    
    def result(self) -> float:
        return self.correct / self.num_samples
