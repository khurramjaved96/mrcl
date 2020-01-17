
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
import numpy as np

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

    tasks = list(range(4000))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", in_channels=args["capacity"] + 1, num_actions=1,
                                       width=args["width"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearnerRegression(args, config).to(device)
    old_model = maml.net
    # maml.net = net = torch.load("/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/1_1/net.model", map_location="cpu")
    maml.net = net = torch.load("/Volumes/Macintosh HD/Users/khurramjaved96/2020Git/small_net_0_15.model",
                                map_location="cpu")
    for old, new in zip(old_model.parameters(), maml.net.parameters()):
        new.learn = old.learn
        print("Learn = ", new.learn)
    #
    # for name, param in maml.named_parameters():
    #     param.learn = True
    #     # if "meta" in name:
    #     #     param.learn = False
    # for name, param in maml.net.named_parameters():
    #     param.learn = True
    #     # if "meta" in name:
    #     #     param.learn = False
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
    #
    # for name, param in maml.named_parameters():
    #     logger.info(name)
    #     if name in frozen_layers:
    #         logger.info("Freeezing name %s", str(name))
    #         param.learn = False
    #         logger.info(str(param.requires_grad))
    #
    #     if "bn" in name:
    #         logger.info("Freeezing name %s", str(name))
    #         param.learn = False
    #
    #
    # for name, param in maml.net.named_parameters():
    #     logger.info(name)
    #     if name in frozen_layers:
    #         logger.info("Freeezing name %s", str(name))
    #         param.learn = False
    #         logger.info(str(param.requires_grad))
    #     if "bn" in name:
    #         logger.info("Freeezing name %s", str(name))
    #         param.learn = False

    old_net = copy.deepcopy(maml.net)

    for lrs in [0.003]:
        lr_results = {}
        lr_results[lrs] = []
        for temp in range(0, 1):
            t1 = np.random.choice(tasks, args["tasks"], replace=False)
            iterators = []
            #
            for t in t1:
                iterators.append(sampler.sample_task([t]))
            x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=10)
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
                # else:
                #     grad = torch.autograd.grad(loss, net.parameters())

                import matplotlib.pyplot as  plt
                for (name, param) in net.named_parameters():

                    if "meta" in name and k ==0:

                        # print(param)
                        histo_plas = torch.sigmoid(param).detach().numpy()
                        # if len(histo_plas.shape) > 1:
                        #     # print(histo_plas.shape)
                        #     plt.imshow(histo_plas)
                        #     plt.colorbar()
                        #     plt.show()
                        #     plt.clf()


                        histo_plas = torch.sigmoid(param).flatten().detach().numpy()
                        plt.hist(histo_plas)
                        plt.title(name)
                        plt.show()
                        plt.clf()
                        #
                        # plt.savefig(my_experiment.path+name+".pdf", format="pdf")
                        # plt.clf()
                0/0
                fast_weights = []
                counter = 0
                for g, (name, p) in zip(grad, net.named_parameters()):
                    if p.learn:
                        # print("Name = ", name)
                        # print("Counter = ", counter)

                        mask = net.meta_vars[counter]
                        # print(g.shape, mask.shape)
                        temp_weight = p - lrs * g * torch.sigmoid(mask)
                        plasticity_plot = torch.sigmoid(mask).detach().numpy()
                        weight_plot = p.detach().numpy()
                        gradient_plot = g.detach().numpy()
                        if len(weight_plot.shape) > 1:
                            plt.imshow(plasticity_plot)
                            plt.title("Plasticity "+name + str(counter))
                            plt.colorbar()
                            plt.show()
                            plt.clf()
                            plt.imshow(weight_plot)
                            plt.title("Weight " + name + str(counter))
                            plt.colorbar()
                            plt.show()
                            plt.clf()
                            plt.imshow(gradient_plot)
                            plt.title("Gradient " + name + str(counter))
                            plt.colorbar()
                            plt.show()
                            plt.clf()

                            plt.imshow(gradient_plot*plasticity_plot)
                            plt.title("Effective gradient" + name + str(counter))
                            plt.colorbar()
                            plt.show()
                            plt.clf()

                        counter += 1
                        if counter>6 or temp < 1 or True:
                            p.data = temp_weight
                    else:
                        temp_weight = p
                    fast_weights.append(temp_weight)

                # 0/0
            0/0
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
            print("diff = ", torch.mean(torch.abs(new- old)))


        logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))

        # torch.save(maml.net, my_experiment.path + "learner.model")



#

if __name__ == '__main__':
    main()
