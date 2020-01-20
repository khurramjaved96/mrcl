import random
from collections import namedtuple
import torch
import logging
import numpy as np

logger = logging.getLogger("experiment")
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')


class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]


class ReservoirSampler:
    def __init__(self, windows, buffer_size=5000):
        self.buffer = []
        self.location = 0
        self.buffer_size = buffer_size
        self.window = windows
        self.total_additions = 0

    def add(self, *args):
        self.total_additions += 1
        stuff_to_add = transition(*args)

        M = len(self.buffer)
        if M < self.buffer_size:
            self.buffer.append(stuff_to_add)
        else:
            i = random.randint(0, min(self.total_additions, self.window))
            if i < self.buffer_size:
                self.buffer[i] = stuff_to_add
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size, er_meta=False):
        if er_meta:
            return random.sample(self.buffer, batch_size)
        else:
            initial_index = random.randint(0, len(self.buffer) - batch_size)
            return self.buffer[initial_index: initial_index + batch_size]


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)
#
#
# def compute_sparisty(r_buffer, device, meta_learner):
#     sample_sparisty = r_buffer.sample(2048)
#
#     states, actions, rewards, dones, next_states = unpack_batch(batch)
#
#     states_v = torch.tensor(states)
#
#     sample_sparisty = transition(*zip(*sample_sparisty))
#     states = torch.cat(sample_sparisty.state).to(device)
#     feature_map = meta_learner.net(states, feature=True)
#     feature_map_temp = (feature_map > 0).float().sum() / np.prod(feature_map.shape)
#     # logger.info("Feature sparsity = %f percent", feature_map_temp * 100)
#
#     alive = (feature_map.sum(dim=0) > 0).float().sum() / feature_map.shape[1]
#     # logger.info("Alive = %f percent", alive * 100)
#
#     logger.info("Sparsity %f | Alive %f", feature_map_temp * 100, alive * 100)
#
def compute_sparisty(r_buffer, device, meta_learner):
    sample_sparisty = r_buffer.sample(2048)

    states, actions, rewards, dones, next_states = unpack_batch(sample_sparisty)

    states = torch.tensor(states).to(device)

    feature_map = meta_learner.net(states, feature=True)
    feature_map_temp = (feature_map > 0).float().sum() / np.prod(feature_map.shape)
    # logger.info("Feature sparsity = %f percent", feature_map_temp * 100)

    alive = (feature_map.sum(dim=0) > 0).float().sum() / feature_map.shape[1]
    # logger.info("Alive = %f percent", alive * 100)

    logger.info("Sparsity %f | Alive %f", feature_map_temp * 100, alive * 100)

def get_luminance(frame):
    frame = 0.2126*frame[:,0,:,:] + 0.7152*frame[:,1,:,:] + 0.0722*frame[:,2,:,:]
    return frame