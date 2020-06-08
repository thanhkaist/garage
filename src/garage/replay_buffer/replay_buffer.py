"""This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.

"""

import abc
from abc import abstractmethod


class ReplayBuffer(metaclass=abc.ABCMeta):
    """Abstract class for Replay Buffer.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        capacity_in_transitions (int): capacity of the replay buffer,
            measured in transitions.
    """

    def __init__(self, env_spec, capacity_in_transitions):
        self.capacity_in_transitions = capacity_in_transitions
        self.env_spec = env_spec
        self._current_size = 0
        self._current_ptr = 0
        self._n_transitions_stored = 0
        self._buffer = {}

    @abstractmethod
    def sample_transitions(self, batch_size):
        """Sample N transitions, where N=batch_size.

        Args:
            batch_size (int): The number of transitions to be sampled.

        Returns:
            SampleBatch: batch_size transitions sampled.
        """

    def sample_path(self):
        """Sample one path of arbitrary length.

        Returns:
            SampleBatch: Batch of transitions that make up a path.

        Raises:
            NotYetImplementedError: If the replay buffer does not support
                path sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict[str, np.ndarray] or TrajectoryBatch): Path to be added to the buffer.
        """

    @abstractmethod
    def add_transitions(self, transitions):
        """Add a batch of transtions to the buffer.

        Args:
            transitions (dict[str, np.ndarray] or TrajectoryBatch): transitions to be added to the buffer.

        """

    @property
    def full(self):
        """bool: True if the buffer has reached its maximum capacity."""
        return self._n_transitions_stored == self.capacity_in_transitions

    @property
    def n_transitions_stored(self):
        """int: Number of transitions currently stored in the buffer."""
        return self._n_transitions_stored

    @property
    def n_paths_stored(self):
        """int: Number of paths currently in the buffer."""
        raise NotImplementedError
