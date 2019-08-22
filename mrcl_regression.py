import argparse
import copy
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F

import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearnerRegression

logger = logging.getLogger('experiment')


def construct_set(iterators, sampler, steps):
    x_traj = []
    y_traj = []
    list_of_ids = list(range(sampler.capacity - 1))

    start_index = 0

    for id, it1 in enumerate(iterators):
        for inner in range(steps):
            x, y = sampler.sample_batch(it1, list_of_ids[(id + start_index) % len(list_of_ids)], 32)
            x_traj.append(x)
            y_traj.append(y)
    #

    x_rand = []
    y_rand = []
    for id, it1 in enumerate(iterators):
        x, y = sampler.sample_batch(it1, list_of_ids[(id + start_index) % len(list_of_ids)], 32)
        x_rand.append(x)
        y_rand.append(y)

    x_rand = torch.stack([torch.cat(x_rand)])
    y_rand = torch.stack([torch.cat(y_rand)])

    x_traj = torch.stack(x_traj)
    y_traj = torch.stack(y_traj)

    return x_traj, y_traj, x_rand, y_rand


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    print(args)

    tasks = list(range(400))
    logger = logging.getLogger('experiment')

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args.capacity + 1)

    config = mf.ModelFactory.get_model(args.model, "Sin", in_channels=args.capacity + 1, num_actions=1,
                                       width=args.width)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearnerRegression(args, config).to(device)

    for name, param in maml.named_parameters():
        param.learn = True
    for name, param in maml.net.named_parameters():
        param.learn = True
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info(maml)
    logger.info('Total trainable tensors: %d', num)
    #
    accuracy = 0

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("net.vars." + str(temp))
    logger.info("Frozen layers = %s", " ".join(frozen_layers))
    for step in range(args.epoch):

        if step == 0:
            for name, param in maml.named_parameters():
                logger.info(name)
                if name in frozen_layers:
                    logger.info("Freeezing name %s", str(name))
                    param.learn = False
                    logger.info(str(param.requires_grad))

            for name, param in maml.net.named_parameters():
                logger.info(name)
                if name in frozen_layers:
                    logger.info("Freeezing name %s", str(name))
                    param.learn = False
                    logger.info(str(param.requires_grad))

        t1 = np.random.choice(tasks, args.tasks, replace=False)

        iterators = []
        for t in t1:
            # print(sampler.sample_task([t]))
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args.update_step)

        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()
        # print(x_spt, y_spt)
        accs = maml(x_traj, y_traj, x_rand, y_rand)
        maml.meta_optim.step()

        if step in [0, 2000, 3000, 4000]:
            for param_group in maml.optimizer.param_groups:
                logger.info("Learning Rate at step %d = %s", step, str(param_group['lr']))

        accuracy = accuracy * 0.95 + 0.05 * accs[-1]
        if step % 5 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            writer.add_scalar('/metatrain/train/runningaccuracy', accuracy, step)
            logger.info("Running average of accuracy = %s", str(accuracy))
            logger.info('step: %d \t training acc (first, last) %s', step, str(accs[0]) + "," + str(accs[-1]))

        if step % 100 == 0:
            counter = 0
            for name, _ in maml.net.named_parameters():
                counter += 1

            for lrs in [args.update_lr]:
                lr_results = {}
                lr_results[lrs] = []
                for temp in range(0, 20):
                    t1 = np.random.choice(tasks, args.tasks, replace=False)
                    iterators = []
                    #
                    for t in t1:
                        iterators.append(sampler.sample_task([t]))
                    x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=40)
                    if torch.cuda.is_available():
                        x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

                    net = copy.deepcopy(maml.net)
                    net = net.to(device)
                    for params_old, params_new in zip(maml.net.parameters(), net.parameters()):
                        params_new.learn = params_old.learn

                    list_of_params = list(filter(lambda x: x.learn, net.parameters()))

                    optimizer = optim.SGD(list_of_params, lr=lrs)
                    for k in range(len(x_traj)):
                        logits = net(x_traj[k], None, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_traj[k, :, 1].long()):
                            logits_select.append(logits[no, val])

                        logits = torch.stack(logits_select).unsqueeze(1)

                        loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    #
                    with torch.no_grad():
                        logits = net(x_rand[0], vars=None, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_rand[0, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                        lr_results[lrs].append(loss_q.item())

                logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))

            torch.save(maml.net, my_experiment.path + "learner.model")


#

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=1000)
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--capacity', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.003)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=40)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_regression")
    argparser.add_argument('--model', help='Name of model', default="old")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--width", type=int, default=300)
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()
    #

    args.name = "/".join(["sin", str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
