from unittest.mock import Mock

import numpy as np
import ray

from garage.envs.grid_world_env import GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.sampler import OnPolicyVectorizedSampler, RaySampler, SamplerWorker
from garage.tf.envs import TfEnv


class TestSampler:
    """
    Uses mock policy for 4x4 gridworldenv
    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]
    0: left
    1: down
    2: right
    3: up
    -1: no move
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    [2,2,1,0,3,1,1,1,2,2,1,1,1,2,2,1]
    """

    def setup_method(self):
        ray.init(local_mode=True, ignore_reinit_error=True)

        self.env = TfEnv(GridWorldEnv(desc='4x4'))
        self.policy = ScriptedPolicy(
            scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
        self.algo = Mock(env_spec=self.env.spec,
                         policy=self.policy,
                         max_path_length=16)

    def teardown_method(self):
        self.env.close()

    def test_ray_batch_sampler(self):
        sampler1 = RaySampler(self.algo,
                              self.env,
                              seed=100,
                              num_processors=1,
                              sampler_worker_cls=SamplerWorker)
        sampler1.start_worker()
        sampler2 = OnPolicyVectorizedSampler(self.algo, self.env)
        sampler2.start_worker()
        trajs1 = sampler1.obtain_samples(0, 16)
        trajs2 = sampler2.obtain_samples(0, 1)
        assert (trajs1[0]['observations'].shape == np.array(
            trajs2[0]['observations']).shape == (6, ))
        traj2_action_shape = np.array(trajs2[0]['actions']).shape
        assert trajs1[0]['actions'].shape == traj2_action_shape == (6, )
        assert sum(trajs1[0]['rewards']) == sum(trajs2[0]['rewards']) == 1

        true_obs = np.array([0, 1, 2, 6, 10, 14])
        true_actions = np.array([2, 2, 1, 1, 1, 2])
        true_rewards = np.array([0, 0, 0, 0, 0, 1])
        for trajectory in trajs1:
            assert np.array_equal(trajectory['observations'], true_obs)
            assert np.array_equal(trajectory['actions'], true_actions)
            assert np.array_equal(trajectory['rewards'], true_rewards)
        sampler1.shutdown_worker()
        sampler2.shutdown_worker()

    def test_ray_sampler_idle_workers(self):
        """Addresses issue #1349

        There is an edge use case where samples aren't collected from workers
        because `num_samples` have already been collected from some subset of
        the sampler workers, then the other workers should have their calls
        to rollout canceled. At the moment, the fix for this is to only call
        rollout on workers if rollout should be called on them. For example
        If there are 2 workers, and they are supposed to each collect 16
        samples and then a call to obtain samples is made asking for only 16
        samples, then only the first sampler worker will be started. The best
        way for us to handle in the future is to shutdown workers using some
        ipc, but handling starting workers the way that is mentioned above
        is the best fix for a backport.

        """
        sampler = RaySampler(self.algo,
                             self.env,
                             seed=100,
                             num_processors=2,
                             sampler_worker_cls=SamplerWorker)
        sampler.start_worker()
        assert len(sampler._idle_worker_ids) == 2
        sampler.obtain_samples(0, 16)
        assert len(sampler._idle_worker_ids) == 2
        sampler.shutdown_worker()
