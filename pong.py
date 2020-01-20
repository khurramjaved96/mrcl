#!/usr/bin/env python3
import argparse

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import model.meta_learner as ml
import model.modelfactory as mf
import ptann as ptan
import trainer.dqn_trainer as trainer
from experiment.experiment import experiment
from lib import common, atari_wrappers
from utils import rl_utils
from utils import utils
import configs.atari.atari_parser as atari_parser

PLAY_STEPS = 4

import logging

logger = logging.getLogger("experiment")


def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True)
    return env


def play_func(params, net, cuda, exp_queue, args):
    device = torch.device("cuda" if cuda else "cpu")
    env = make_env(params)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-05_new_wrappers")
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    my_experiment = experiment(args["name"] + "_eval", args, "../results/", commit_changes=False)
    my_experiment.store_json()

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

    exp_queue.put(None)

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))




if __name__ == "__main__":
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['pong']
    print(params)
    # 0/0
    warnings.showwarning = warn_with_traceback
    params['batch_size'] *= PLAY_STEPS

    p = atari_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False)
    my_experiment.store_json()

    env = make_env(params)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        args["cuda"] = True
    else:
        device = torch.device('cpu')
        args["cuda"] = False

    num_actions = env.action_space.n

    in_channels = env.observation_space.shape[0]

    # logger.info("In channels = %s", str(env.observation_space.shape))
    policy_net_config = mf.ModelFactory.get_model('na', 'atari', in_channels, num_actions)
    meta_learner = ml.MetaRL2(args, policy_net_config).to(device)

    # Freezing RLN layers

    net = meta_learner.net
    for (name, param) in net.named_parameters():
        print(name, param.learn)
    # 0/0
    # net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    print("Length of params = ", len(net.parameters()))
    print("Length of params after = ", len(list(filter(lambda x : x.learn, net.parameters()))))
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(list(filter(lambda x : x.learn, net.parameters())), lr=params['learning_rate'])

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args["cuda"], exp_queue, args))
    play_proc.start()

    frame_idx = 0
    t = 0
    while play_proc.is_alive():
        t += 1
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                      cuda=args["cuda"], cuda_async=True)
        loss_v.backward()
        optimizer.step()

        # META LEARNING CODE
        #
        # if len(buffer) > 1.2 * params['replay_initial'] and not args["metaoff"] and t % args["freq"] == 0:
        #     for _ in range(100):
        #         sample_rand = buffer.sample(1024)
        #         if False and args["maml"]:
        #             sample_traj = buffer.sample(200)
        #         else:
        #             sample_traj = buffer.sample_trajectory(100)
        #         trainer.train_meta(sample_traj, meta_learner, tgt_net.target_model, sample_rand)
        #
        # if len(buffer.buffer) > 3000 and t % 1000 == 0:
        #     rl_utils.compute_sparisty(buffer, device, meta_learner)

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()