import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.regression.reg_parser_inspect as reg_parser
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

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print(args)

    tasks = list(range(4000))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", input_dimension=args["capacity"] + 1, output_dimension=1,
                                       width=args["width"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    metalearner = MetaLearnerRegression(args, config).to(device)

    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Plot/mrcl_meta_sgd_parameter_sensitivity/0/1_1"
    # base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Plot/meta_sgd_parameter_sensitivity/0/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Other/meta_sgd_warped_parameter_sensitivity/6/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Other/plasticity_warped_parameter_sensitivity/20/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Other/mrcl_meta_sgd_parameter_sensitivity/6/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Plot/meta_sgd_parameter_sensitivity/6/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Plot/meta_sgd_parameter_sensitivity/2/1_1"
    base = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Comparisons/Parameter_senstivity/Plot/mrcl_meta_sgd_parameter_sensitivity/2/1_1"
    base = args["base"]
    metalearner.net = net = torch.load(base + "/net.model",
                                map_location="cpu").to(device)
    import json
    with open(base + "/metadata.json") as json_file:
        data = json.load(json_file)
    layers_learn = data["results"]["Layers meta values"]
    for (name, param) in metalearner.net.named_parameters():
        if name in layers_learn:
            param.learn = layers_learn[name]
            print(name, layers_learn[name])


    lrs = data["params"]["update_lr"]
    plasticity = not data["params"]["no_plasticity"]
    sigmoid = not data["params"]["no_sigmoid"]
    old_net = copy.deepcopy(metalearner.net)

    lr_results = {}
    lr_results[lrs] = []
    for temp in range(0, 30):
        t1 = np.random.choice(tasks, args["tasks"], replace=False)
        iterators = []
        #
        for t in t1:
            iterators.append(sampler.sample_task([t]))
        x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=10)
        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

        net = metalearner.net

        for k in range(len(x_traj)):

            logits = net(x_traj[k], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss_temp = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))


            grad = metalearner.clip_grad(torch.autograd.grad(loss_temp, list(filter(lambda x: x.learn, list(net.parameters())))))

            grad = metalearner.clip_grad(
                torch.autograd.grad(loss_temp, list(filter(lambda x: x.learn, list(net.parameters())))))

            list_of_context = None
            if metalearner.context:
                list_of_context = metalearner.net.forward_plasticity(x_traj[k])

            metalearner.inner_update(net, grad, adaptation_lr, list_of_context)

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

    for (name, old), new in zip(old_net.named_parameters(), net.parameters()):
        print("Name = ", name)
        print("diff = ", torch.mean(torch.abs(new - old)))

    logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))

    # torch.save(maml.net, my_experiment.path + "learner.model")


#

if __name__ == '__main__':
    main()
