import logging
from collections import namedtuple

import torch
import torch.nn.functional as f

logger = logging.getLogger('experiment')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

GAMMA = 0.99


def train(sample, policy_net, target_net, optimizer):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))

    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

    # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
    # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
    # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
    # Q_s_a is of size (BATCH_SIZE, 1).
    Q_s_a = policy_net(states.to(device)).gather(1, actions)

    # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
    # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
    # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
    # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0],
                                                  dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        Q_s_prime_a_prime[none_terminal_next_state_index] = \
            target_net(none_terminal_next_states.to(device)).detach().max(1)[
                0].unsqueeze(1).to(device)

    # Compute the target
    target = rewards + GAMMA * Q_s_prime_a_prime

    # Huber loss
    loss = f.smooth_l1_loss(target.to(device), Q_s_a)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#     Do the meta learning update now
import numpy as np


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def train_meta(data_traj, meta_learner, target_net, data_rand):
    #######################################################
    #### GETTING TARGETS FOR DATA IN THE TRAJECTORY SAMPLE ####
    #######################################################

    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys

    states_traj, actions_traj, rewards_traj, dones_traj, next_states_traj = unpack_batch(data_traj)

    states_traj = torch.tensor(states_traj).to(device)
    next_states_traj = torch.tensor(next_states_traj).to(device)
    actions_traj = torch.tensor(actions_traj).to(device)
    rewards_traj = torch.tensor(rewards_traj).to(device)
    is_terminal_traj = torch.ByteTensor(dones_traj).to(device).bool()

    next_state_values_traj = target_net(next_states_traj).max(1)[0]
    next_state_values_traj[is_terminal_traj] = 0.0

    target_traj = next_state_values_traj.detach() * GAMMA + rewards_traj

    #######################################################
    #### GETTING TARGETS FOR DATA IN THE RANDOM SAMPLE ####
    #######################################################

    states_rand, actions_rand, rewards_rand, dones_rand, next_states_rand = unpack_batch(data_rand
                                                                                         )
    states_rand = torch.tensor(states_rand).to(device)
    next_states_rand = torch.tensor(next_states_rand).to(device)
    actions_rand = torch.tensor(actions_rand).to(device)
    rewards_rand = torch.tensor(rewards_rand).to(device)
    is_terminal_rand = torch.ByteTensor(dones_rand).to(device).bool()

    next_state_values_rand = target_net(next_states_rand).max(1)[0]
    next_state_values_rand[is_terminal_rand] = 0.0

    target_test = next_state_values_rand.detach() * GAMMA + rewards_rand

    # logger.info("State_traj %s, target traj %s, action_traj %s, states_rand %s, target_rand %s, actions_rand %s",
    #             str(states_traj.shape), str(target_traj.shape), str(actions_traj.shape), str(states_rand.shape),
    #             str(target_test.shape), str(actions_rand.shape))

    Q_s_a = meta_learner(states_traj.unsqueeze(1).to(device), target_traj.unsqueeze(1).to(device),
                         actions_traj.unsqueeze(1).to(device),
                         states_rand.unsqueeze(0).to(device), target_test.unsqueeze(0).to(device),
                         actions_rand.unsqueeze(0).to(device))

    return Q_s_a[-1]


def train_meta_2(data_traj, meta_learner, target_net, data_rand):
    #######################################################
    #### GETTING TARGETS FOR DATA IN THE TRAJECTORY SAMPLE ####
    #######################################################

    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples_traj = transition(*zip(*data_traj))
    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states_traj = torch.cat(batch_samples_traj.state)
    next_states_traj = torch.cat(batch_samples_traj.next_state)
    actions_traj = torch.cat(batch_samples_traj.action)
    rewards_traj = torch.cat(batch_samples_traj.reward)
    is_terminal_traj = torch.cat(batch_samples_traj.is_terminal).bool()

    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index_traj = torch.tensor(
        [i for i, is_term in enumerate(is_terminal_traj) if is_term == 0],
        dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states_traj = next_states_traj.index_select(0, none_terminal_next_state_index_traj)

    Q_s_prime_a_prime_traj = torch.zeros(len(data_traj), 1, device=device)
    if len(none_terminal_next_states_traj) != 0:
        Q_s_prime_a_prime_traj[none_terminal_next_state_index_traj] = \
            target_net(none_terminal_next_states_traj).detach().max(1)[
                0].unsqueeze(1)

    # Compute the target
    target_traj = rewards_traj + GAMMA * Q_s_prime_a_prime_traj

    #######################################################
    #### GETTING TARGETS FOR DATA IN THE RANDOM SAMPLE ####
    #######################################################

    batch_samples_rand = transition(*zip(*data_rand))
    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states_rand = torch.cat(batch_samples_rand.state)
    next_states_rand = torch.cat(batch_samples_rand.next_state)
    actions_rand = torch.cat(batch_samples_rand.action)
    rewards_rand = torch.cat(batch_samples_rand.reward)
    is_terminal_rand = torch.cat(batch_samples_rand.is_terminal).bool()

    # Get the indices of next_states that are not terminal
    none_terminal_next_state_index_rand = torch.tensor(
        [i for i, is_term in enumerate(is_terminal_rand) if is_term == 0],
        dtype=torch.int64, device=device)
    # Select the indices of each row
    none_terminal_next_states_rand = next_states_rand.index_select(0, none_terminal_next_state_index_rand)

    Q_s_prime_a_prime_rand = torch.zeros(len(data_rand), 1, device=device)
    if len(none_terminal_next_states_rand) != 0:
        Q_s_prime_a_prime_rand[none_terminal_next_state_index_rand] = \
            target_net(none_terminal_next_states_rand).detach().max(1)[
                0].unsqueeze(1)

    #
    #
    #
    #
    # Compute the target
    target_test = rewards_rand + GAMMA * Q_s_prime_a_prime_rand

    # Huber loss

    Q_s_a = meta_learner(states_traj.unsqueeze(1).to(device), target_traj.unsqueeze(1).to(device),
                         actions_traj.unsqueeze(1).to(device),
                         states_rand.unsqueeze(0).to(device), target_test.unsqueeze(0).to(device),
                         actions_rand.unsqueeze(0).to(device))

    return Q_s_a[-1]
