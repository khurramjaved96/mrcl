import copy
import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.regression.reg_parser as reg_parser
import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearnerRegression
from utils import utils

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
    # print(x_traj.shape, y_traj.shape, x_rand.shape, y_rand.shape)
    # 0/0
    return x_traj, y_traj, x_rand, y_rand


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=args["commit"],
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print(args)

    tasks = list(range(400))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", in_channels=args["capacity"] + 1, num_actions=1,
                                       width=args["width"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearnerRegression(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # logger.info(maml)
    logger.info('Total trainable tensors: %d', num)
    #
    accuracy = 0

    # frozen_layers = []
    # for temp in range(args["rln"] * 2):
    #     frozen_layers.append("net.vars." + str(temp))
    # logger.info("Frozen layers = %s", " ".join(frozen_layers))
    for step in range(args["epoch"]):
    #
        if step == 0:
            for name, param in maml.named_parameters():
                logger.info("Name = %s", name)
                logger.info("Learn = %s", str(param.learn))
    #             if name in frozen_layers:
    #                 logger.info("Freeezing name %s", str(name))
    #                 param.learn = False
    #                 logger.info(str(param.requires_grad))
    #
    #             if "bn" in name:
    #                 logger.info("Freeezing name %s", str(name))
    #                 param.learn = False
    #
    #
    #         for name, param in maml.net.named_parameters():
    #             logger.info(name)
    #             if name in frozen_layers:
    #                 logger.info("Freeezing name %s", str(name))
    #                 param.learn = False
    #                 logger.info(str(param.requires_grad))
    #             if "bn" in name:
    #                 logger.info("Freeezing name %s", str(name))
    #                 param.learn = False

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            # print(sampler.sample_task([t]))
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args["update_step"])

        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()
        # print(x_spt, y_spt)
        accs = maml(x_traj, y_traj, x_rand, y_rand)
        maml.meta_optim.step()
        # maml.meta_optim_plastic.step()

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

            for lrs in [args["update_lr"]]:
                lr_results = {}
                lr_results[lrs] = []
                for temp in range(0, 10):
                    t1 = np.random.choice(tasks, args["tasks"], replace=False)
                    iterators = []
                    #
                    for t in t1:
                        iterators.append(sampler.sample_task([t]))
                    x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args["update_step"])
                    if torch.cuda.is_available():
                        x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

                    net = copy.deepcopy(maml.net)

                    # net = torch.load("/Volumes/Macintosh HD/Users/khurramjaved96/ICML_2020/pretrained_plasticity5/1_1/net.model", map_location="cpu")
                    for params_old, params_new in zip(maml.net.parameters(), net.parameters()):
                        params_new.learn = params_old.learn

                    for (name, p) in net.named_parameters():
                        if "meta" in name:
                            p.learn = False



                    for k in range(len(x_traj)):

                        logits = net(x_traj[k], vars=None, bn_training=False)
                        logits_select = []
                        for no, val in enumerate(y_traj[k, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                        # if k < 10:
                        grad = maml.clip_grad(torch.autograd.grad(loss, net.parameters()))

                        fast_weights = []
                        counter = 0

                        for g, (name, p) in zip(grad, net.named_parameters()):
                            if p.learn:
                                # print("Name = ", name)
                                # print("Counter = ", counter)

                                mask = net.meta_vars[counter]
                                # print(g.shape, mask.shape)
                                temp_weight = p - lrs * g * torch.sigmoid(mask)
                                counter += 1
                                if counter > 6 or temp < 1 or True:
                                    p.data = temp_weight
                            else:
                                temp_weight = p
                            fast_weights.append(temp_weight)

                    #
                    with torch.no_grad():
                        logits = net(x_rand[0], vars=None, bn_training=False)
                        # print("Logits = ", logits)
                        logits_select = []
                        for no, val in enumerate(y_rand[0, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        # print("Logits = ", logits)
                        # print("Targets = ", y_rand[0, :, 0].unsqueeze(1))
                        loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                        print(loss_q)
                        lr_results[lrs].append(loss_q.item())


                logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))
        if step % 100 == 0:
            torch.save(maml.net, my_experiment.path + "net.model")
            dict_names = {}
            for (name, param) in maml.net.named_parameters():
                dict_names[name] = param.learn
            my_experiment.add_result("Layers meta values", dict_names)
            my_experiment.store_json()
            # torch.save(maml.meta_net, my_experiment.path + "meta_net.model")


#

if __name__ == '__main__':
    main()
