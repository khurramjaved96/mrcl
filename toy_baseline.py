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


def construct_set(iterators, sampler, steps=2, offset=0):
    x_spt = []
    y_spt = []
    for id, it1 in enumerate(iterators):
        for inner in range(steps):

            x, y = sampler.sample_batch(it1, id+offset, 32)
            # print(x.shape, y.shape)
            x_spt.append(x)
            y_spt.append(y)

    x_qry = []
    y_qry = []
    for id, it1 in enumerate(iterators):
        x, y = sampler.sample_batch(it1, id+offset, 32)
        x_qry.append(x)
        y_qry.append(y)

    x_qry = torch.stack([torch.cat(x_qry)])
    y_qry = torch.stack([torch.cat(y_qry)])

    x_spt = torch.stack(x_spt)
    y_spt = torch.stack(y_spt)

    return x_spt, y_spt, x_qry, y_qry


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    print(args)

    tasks = list(range(400))
    logger = logging.getLogger('experiment')

    sampler = ts.SamplerFactory.get_sampler("Sin2", tasks, None, None, capacity=401)

    config = mf.ModelFactory.get_model("na", "Sin", in_channels=11, num_actions=30, width=args.width)

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
    for temp in range(args.total_frozen * 2):
        frozen_layers.append("net.vars." + str(temp))
    logger.info("Frozen layers = %s", " ".join(frozen_layers))

    opt = torch.optim.Adam(maml.parameters(), lr=args.lr)
    meta_optim = torch.optim.lr_scheduler.MultiStepLR(opt, [5000, 8000], 0.2)

    for step in range(args.epoch):
        if step %300 == 0:
            print(step)
        for heads in range(30):
            t1 = tasks
            # print(tasks)
            iterators = []
            if not args.baseline:
                for t in range(heads*10, heads*10+10):
                    # print(sampler.sample_task([t]))
                    # print(t)
                    iterators.append(sampler.sample_task([t]))

            else:
                iterators.append(sampler.get_another_complete_iterator())

            x_spt, y_spt, x_qry, y_qry = construct_set(iterators, sampler, steps=args.update_step, offset =heads*10)

            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
            # print(x_spt, y_spt)
            net = maml.net
            logits = net(x_qry[0], None, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_qry[0, :, 1].long()):
                # print(y_qry[0, :, 1].long())
                logits_select.append(logits[no, val])

            logits = torch.stack(logits_select).unsqueeze(1)

            loss = F.mse_loss(logits, y_qry[0, :, 0].unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            meta_optim.step()
            # print(loss)
            accuracy = accuracy * 0.95 + 0.05 * loss
            if step % 500 == 0:
                writer.add_scalar('/metatrain/train/accuracy', loss, step)
                writer.add_scalar('/metatrain/train/runningaccuracy', accuracy, step)
                logger.info("Running average of accuracy = %s", str(accuracy.item()))

            if step%500 == 0:
                torch.save(maml.net, my_experiment.path + "learner.model")


# #

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--classes', type=int, nargs='+', help='Total classes to use in training',
                           default=[0, 1, 2, 3, 4])
    argparser.add_argument("--baseline", "-b", action="store_true")
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.003)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="dolphin")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument("--width", type=int, default=300)
    argparser.add_argument("--capacity", type=int, default=400)
    argparser.add_argument("--total-frozen", type=int, default=4)
    args = argparser.parse_args()

    import os


    args.name = "/".join(["sin", "baseline", args.name])
    print(args)
    main(args)
