import numpy as np

from garage.replay_buffer.path_buffer import PathBuffer


class PartialPathBuffer:

    def __init__(self, capacity_in_transitions):
        self._completed_buffer = PathBuffer(capacity_in_transitions)
        self._partial_paths = {}
        self._transitions_stored = 0

    def add_samples(self, samples, path_keys, path_ends):
        """Add a batch of samples from different paths.

        Args:
            samples (dict[str, np.ndarray]): Samples to add to paths.
            path_keys (list[int]): List identify which path each sample is
                from. Must have the same length as the arrays in samples and as
                path_ends. Keys must be unique for each path for the duration
                that the path is being gathered. However, when the path ends
                its key can be re-used. If samples are being gathered in
                parallel and consistently packed, then a constant array can be
                used. E.g.  [0, 1, 2, 3].
            path_ends (list[bool]): Indicates if each sample should be
                considered the end of a path.

        Raises:
            ValueError: if the
        """
        n_samples = len(path_keys)
        self._transitions_stored += n_samples
        if n_samples != len(path_ends):
            raise ValueError('Wrong number of path ends.')
        for i in range(len(path_keys)):
            path_key = path_keys[i]
            end = path_ends[i]
            path = self._partial_paths[path_key]
            for arr_key, arr in samples.items():
                if len(arr) != n_samples:
                    raise ValueError('Wrong number of {!r}s'.format(arr_key))
                v = arr[i]
                path.setdefault(arr_key, []).append(v)
            if end:
                path_len = len(next(path.values()))
                np_path = {}
                for k, v in path.items():
                    assert len(v) == path_len
                    np_path[k] = np.asarray(v)
                self._completed_buffer.add_path(np_path)
                self._transitions_stored -= path_len
                del self._partial_paths[path_key]

    def sample_path(self):
        total_paths = (len(self._partial_paths) +
                       self._completed_buffer.n_paths_stored)
        path_num = np.random.randint(total_paths)
        if path_num < len(self._partial_paths):
            path_key = list(self._partial_paths.keys())[path_num]
            path = self._partial_paths[path_key]
            return {k: np.asarray(v) for (k, v) in path.items()}
        else:
            return self._completed_buffer.sample_path()

    def sample_transitions(self, batch_size):
        total_transitions = (self._transitions_stored +
                             self._completed_buffer.n_transitions_stored)
        samples = {}
        sample_indices = np.random.randint(total_transitions,
                                           size=total_transitions)
        for sample_idx in sample_indices[sample_indices <
                                         self._transitions_stored]:
            for path_key, path in self._partial_paths:
                path_samples = len(next(path.values()))
                if sample_idx < path_samples:
                    for k, v in path:
                        samples.setdefault(k, []).append(v[sample_idx])
                    break
                else:
                    sample_idx -= path_samples
        n = sum(sample_indices >= self._transitions_stored)
        complete_path_samples = self._completed_buffer.sample_transitions(n)
        np_samples = {k: np.concatenate((v, complete_path_samples[k]))
                      for (k, v) in samples}
        shuffle_indices = np.arange(0, batch_size)
        np.random.shuffle(shuffle_indices)
        return {k: v[shuffle_indices] for (k, v) in np_samples.items()}
