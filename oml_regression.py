import copy
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


#

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

    tasks = list(range(400))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", input_dimension=args["capacity"] + 1, output_dimension=1,
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

    for step in range(args["epoch"]):
        #
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

        accs = metalearner(x_traj, y_traj, x_rand, y_rand)

        if step in [0, 2000, 3000, 4000]:
            for param_group in metalearner.optimizer.param_groups:
                logger.info("Learning Rate at step %d = %s", step, str(param_group['lr']))

        accuracy = accuracy * 0.95 + 0.05 * accs[-1]
        if step % 5 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            writer.add_scalar('/metatrain/train/runningaccuracy', accuracy, step)
            logger.info("Running accuracy = %f", accuracy.item())
            logger.debug('Step: %d \t Meta-training loss: First: %f \t Last: %f', step, accs[0].item(), accs[-1].item())

        if step % 100 == 0:
            counter = 0
            for name, _ in metalearner.net.named_parameters():
                counter += 1

            for lrs in [args["update_lr"]]:
                lr_results = {}
                lr_results[lrs] = []
                for temp in range(0, 10):
                    t1 = np.random.choice(tasks, args["tasks"], replace=False)
                    iterators = []

                    for t in t1:
                        iterators.append(sampler.sample_task([t]))
                    x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args["update_step"])
                    if torch.cuda.is_available():
                        x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

                    net = copy.deepcopy(metalearner.net)

                    for params_old, params_new in zip(metalearner.net.parameters(), net.parameters()):
                        params_new.learn = params_old.learn

                    for (name, p) in net.named_parameters():
                        if "meta" in name or "neuro" in name:
                            p.learn = False

                    for k in range(len(x_traj)):

                        logits = net(x_traj[k], vars=None, bn_training=False)
                        logits_select = []
                        for no, val in enumerate(y_traj[k, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                        # if k < 10:
                        grad = metalearner.clip_grad(torch.autograd.grad(loss, net.parameters()))

                        fast_weights = []
                        counter = 0

                        for g, (name, p) in zip(grad, net.named_parameters()):
                            if p.learn:
                                mask = net.meta_plasticity[counter]
                                if not args["no_plasticity"]:
                                    temp_weight = p - lrs * g
                                else:
                                    if args["no_sigmoid"]:
                                        temp_weight = p - lrs * g * mask
                                    else:
                                        temp_weight = p - lrs * g * torch.sigmoid(mask)
                                counter += 1
                                p.data = temp_weight.data
                            else:
                                temp_weight = p
                            fast_weights.append(temp_weight)
                    #
                    with torch.no_grad():
                        logits = net(x_rand[0], vars=None, bn_training=False)
                        logits_select = []
                        for no, val in enumerate(y_rand[0, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                        logger.debug("Loss = %f", loss_q.item())
                        lr_results[lrs].append(loss_q.item())

                logger.warning("Avg MSE LOSS for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))
                writer.add_scalar('/metatest/test/averageaccuracy', np.mean(lr_results[lrs]), step)

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
