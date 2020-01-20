import logging

import numpy as np
import torch
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

    return x_traj, y_traj, x_rand, y_rand


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print("Selected args", args)

    tasks = list(range(2000))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", in_channels=args["capacity"] + 1, num_actions=1,
                                       width=args["width"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    metalearner = MetaLearnerRegression(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, metalearner.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info('Total trainable tensors: %d', num)

    accuracy = 0
    adaptation_accuracy = 0
    for step in range(args["epoch"]):
        if step % 5 == 0:
            logger.warning("####\t STEP %d \t####", step)

        lrs = args["update_lr"]
        if step == 0:
            for name, param in metalearner.named_parameters():
                logger.info("Name = %s, learn = %s", name, str(param.learn))

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args["update_step"])

        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()
        net = metalearner.net
        for k in range(len(x_traj)):

            logits = net(x_traj[k], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
            # if k < 10:

            grad = metalearner.clip_grad(
                torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(net.parameters())))))

            counter = 0
            for (name, p) in net.named_parameters():
                if "meta" in name or "neuro" in name:
                    pass
                if p.learn:
                    g = grad[counter]
                    # print(g)
                    mask = net.meta_plasticity[counter]
                    # print(g.shape, p.shape, mask.shape)
                    if metalearner.plasticity:
                        if metalearner.sigmoid:
                            p.data -= lrs * g * torch.sigmoid(mask)
                        else:
                            p.data -= lrs * g * mask
                    else:
                        p.data -= lrs * g
                    counter += 1
                else:
                    pass

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
            adaptation_accuracy = adaptation_accuracy * 0.85 + loss_q.detach().item() * 0.15
            # adaptation_accuracy /= (1-(0.85**(step+1)))
            if step % 5 == 0:
                logger.info("Running adaptation loss = %f", adaptation_accuracy)
            # logger.info("Adaptation loss = %f", loss_q.item())
            writer.add_scalar('/learn/train/accuracy', loss_q, step)
            # lr_results[lrs].append(loss_q.item())

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args["update_step"])

        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

        accs = metalearner(x_traj, y_traj, x_rand, y_rand)

        metalearner.meta_optim.step()
        if not args["no_plasticity"]:
            metalearner.meta_optim_plastic.step()
        # if not args["no_neuro"]:
        #     metalearner.meta

        if step in [0, 2000, 3000, 4000]:
            for param_group in metalearner.optimizer.param_groups:
                logger.info("Learning Rate at step %d = %s", step, str(param_group['lr']))

        accuracy = accuracy * 0.85 + 0.15 * accs[-1]
        # accuracy /= (1-(0.85**(step+1)))
        writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
        writer.add_scalar('/metatrain/train/runningaccuracy', accuracy, step)

        if step % 5 == 0:
            logger.info("Running meta-loss = %f", accuracy.item())
        if step % 20 == 0:
            logger.debug('Meta-training loss: Before adaptation: %f \t After adaptation: %f', accs[0].item(),
                         accs[-1].item())

        if step % 100 == 0:
            torch.save(metalearner.net, my_experiment.path + "net.model")
            dict_names = {}
            for (name, param) in metalearner.net.named_parameters():
                dict_names[name] = param.learn
            my_experiment.add_result("Layers meta values", dict_names)
            my_experiment.store_json()


#

if __name__ == '__main__':
    main()
