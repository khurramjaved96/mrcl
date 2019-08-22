import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F

import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
import replay as rep
import datasets.miniimagenet as imgnet

logger = logging.getLogger('experiment')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    total_clases = 10

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("vars." + str(temp))
    logger.info("Frozen layers = %s", " ".join(frozen_layers))


    final_results_all = []
    total_clases = args.schedule
    for tot_class in total_clases:
        lr_list = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003]
        for aoo in range(0, args.runs):

            keep = np.random.choice(list(range(20)), tot_class, replace=False)
            #

            dataset = imgnet.MiniImagenet(args.dataset_path, mode='test', elem_per_class=30, classes=keep, seed=aoo)

            dataset_test = imgnet.MiniImagenet(args.dataset_path, mode='test', elem_per_class=30, test=args.test, classes=keep, seed=aoo)

            # Iterators used for evaluation


            iterator = torch.utils.data.DataLoader(dataset_test, batch_size=128,
                                                   shuffle=True, num_workers=1)
            iterator_sorted = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                   shuffle=args.iid, num_workers=1)


            #
            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10
                for lr in lr_list:

                    print(lr)
                    # for lr in [0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
                    maml = torch.load(args.model, map_location='cpu')

                    if args.scratch:
                        config = mf.ModelFactory.get_model("na", args.dataset)
                        maml = learner.Learner(config)
                        # maml = MetaLearingClassification(args, config).to(device).net

                    maml = maml.to(device)

                    for name, param in maml.named_parameters():
                        param.learn = True

                    for name, param in maml.named_parameters():
                        # logger.info(name)
                        if name in frozen_layers:
                            # logger.info("Freeezing name %s", str(name))
                            param.learn = False
                            # logger.info(str(param.requires_grad))
                        else:
                            if args.reset:
                                w = nn.Parameter(torch.ones_like(param))
                                # logger.info("W shape = %s", str(len(w.shape)))
                                if len(w.shape) > 1:
                                    torch.nn.init.kaiming_normal_(w)
                                else:
                                    w = nn.Parameter(torch.zeros_like(param))
                                param.data = w
                                param.learn = True

                    frozen_layers = []
                    for temp in range(args.rln * 2):
                        frozen_layers.append("vars." + str(temp))

                    torch.nn.init.kaiming_normal_(maml.parameters()[-2])
                    w = nn.Parameter(torch.zeros_like(maml.parameters()[-1]))
                    maml.parameters()[-1].data = w

                    for n, a in maml.named_parameters():
                        n = n.replace(".", "_")
                        # logger.info("Name = %s", n)
                        if n == "vars_14":
                            w = nn.Parameter(torch.ones_like(a))
                            # logger.info("W shape = %s", str(w.shape))
                            torch.nn.init.kaiming_normal_(w)
                            a.data = w
                        if n == "vars_15":
                            w = nn.Parameter(torch.zeros_like(a))
                            a.data = w

                    correct = 0

                    for img, target in iterator:
                        with torch.no_grad():
                            img = img.to(device)
                            target = target.to(device)
                            logits_q = maml(img, vars=None, bn_training=False, feature=False)
                            pred_q = (logits_q).argmax(dim=1)
                            correct += torch.eq(pred_q, target).sum().item() / len(img)

                    logger.info("Pre-epoch accuracy %s", str(correct / len(iterator)))

                    filter_list = ["vars.0", "vars.1", "vars.2", "vars.3", "vars.4", "vars.5"]

                    logger.info("Filter list = %s", ",".join(filter_list))
                    list_of_names = list(
                        map(lambda x: x[1], list(filter(lambda x: x[0] not in filter_list, maml.named_parameters()))))

                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                    if args.scratch or args.no_freeze:
                        print("Empty filter list")
                        list_of_params = maml.parameters()
                    #
                    for x in list_of_names:
                        logger.info("Unfrozen layer = %s", str(x[0]))
                    opt = torch.optim.Adam(list_of_params, lr=lr)
                    res_sampler = rep.ReservoirSampler(mem_size)
                    for _ in range(0, args.epoch):
                        for img, y in iterator_sorted:
                            if mem_size > 0:
                                res_sampler.update_buffer(zip(img, y))
                                res_sampler.update_observations(len(img))
                                img = img.to(device)
                                y = y.to(device)
                                img2, y2 = res_sampler.sample_buffer(8)
                                img2 = img2.to(device)
                                y2 = y2.to(device)

                                img = torch.cat([img, img2], dim=0)
                                y = torch.cat([y, y2], dim=0)
                            else:
                                img = img.to(device)
                                y = y.to(device)


                            pred = maml(img)
                            opt.zero_grad()
                            loss = F.cross_entropy(pred, y)
                            loss.backward()
                            opt.step()

                    logger.info("Result after one epoch for LR = %f", lr)
                    correct = 0
                    for img, target in iterator:
                        img = img.to(device)
                        target = target.to(device)
                        logits_q = maml(img, vars=None, bn_training=False, feature=False)

                        pred_q = (logits_q).argmax(dim=1)
                        # print("Pred=", pred_q)
                        # print("Target=", target)
                        correct += torch.eq(pred_q, target).sum().item() / len(img)

                    logger.info(str(correct / len(iterator)))
                    if (correct / len(iterator) > max_acc):
                        max_acc = correct / len(iterator)
                        max_lr = lr

                lr_list = [max_lr]
                results_mem_size[mem_size] = (max_acc, max_lr)
                logger.info("Final Max Result = %s", str(max_acc))
                writer.add_scalar('/finetune/best_' + str(aoo), max_acc, tot_class)
            final_results_all.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Final results = %s", str(results_mem_size))

            my_experiment.results["Final Results"] = final_results_all
            my_experiment.store_json()
            print("FINAL RESULTS = ", final_results_all)
    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--schedule', type=int, nargs='+', default=[2, 4, 6, 8, 10],
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument('--dataset-path', help='Name of experiment', default="imagenet")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--runs", type=int, default=5)

    args = argparser.parse_args()

    import os

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)
