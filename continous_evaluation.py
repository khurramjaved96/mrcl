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
from torchnet.meter import confusionmeter

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

    correct_array = np.zeros(700)
    total_array = np.zeros(700)
    lr_list = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001, 0.0000003, 0.0000001]
    lr_all = []
    for lr_search in range(0, 1):

        keep = np.random.choice(list(range(650)), args.limit, replace=False)
        # np.random.shuffle(keep)

        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args.dataset_path), keep)
        iterator_sorted = torch.utils.data.DataLoader(
            utils.iterator_sorter_omni(dataset, False, classes=total_clases),
            batch_size=1,
            shuffle=args.iid, num_workers=2)
        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=not args.test, background=False, path=args.dataset_path), keep)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               shuffle=False, num_workers=1)


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
                        param.learn = False

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

                for _ in range(0, args.epoch):
                    for img, y in iterator_sorted:
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

                    correct += torch.eq(pred_q, target).sum().item() / len(img)

                logger.info(str(correct / len(iterator)))
                if (correct / len(iterator) > max_acc):
                    max_acc = correct / len(iterator)
                    max_lr = lr

            lr_all.append(max_lr)
            results_mem_size[mem_size] = (max_acc, max_lr)
            logger.info("Final Max Result = %s", str(max_acc))
            writer.add_scalar('/finetune/best_' + str(lr_search), max_acc, args.limit)
        final_results_all.append((args.limit, results_mem_size))
        print("A=  ", results_mem_size)
        logger.info("Temp Results = %s", str(results_mem_size))

        my_experiment.results["Temp Results"] = final_results_all
        my_experiment.store_json()
        print("LR RESULTS = ", final_results_all)

    final_results_all = []

    from scipy import stats
    best_lr = float(stats.mode(lr_all)[0][0])

    logger.info("BEST LR %s= ", str(best_lr))

    for lr_search in range(0, args.runs):

        keep = np.random.choice(list(range(650)), args.limit, replace=False)
        # np.random.shuffle(keep)

        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args.dataset_path), keep)
        iterator_sorted = torch.utils.data.DataLoader(
            utils.iterator_sorter_omni(dataset, False, classes=total_clases),
            batch_size=1,
            shuffle=args.iid, num_workers=2)
        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=not args.test, background=False,
                                          path=args.dataset_path), keep)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               shuffle=False, num_workers=1)

        print(args)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        results_mem_size = {}

        for mem_size in [args.memory]:
            max_acc = -10
            max_lr = -10

            lr = best_lr
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
                    param.learn = False

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

            for _ in range(0, args.epoch):
                for img, y in iterator_sorted:
                    img = img.to(device)
                    y = y.to(device)

                    pred = maml(img)
                    opt.zero_grad()
                    loss = F.cross_entropy(pred, y)
                    loss.backward()
                    opt.step()

            logger.info("Result after one epoch for LR = %f", lr)
            correct = 0
            correct_array = np.zeros(700)
            total_array = np.zeros(700)
            cMatrix = confusionmeter.ConfusionMeter(700, True)
            for img, target in iterator:
                img = img.to(device)
                target = target.to(device)
                logits_q = maml(img, vars=None, bn_training=False, feature=False)

                pred_q = (logits_q).argmax(dim=1)
                print(target.cpu())

                total_array[target.cpu()] += 1
                cMatrix.add(pred_q.squeeze(), target.data.view_as(pred_q).squeeze())

                correct_array[np.nonzero(pred_q.cpu() == target.cpu())] +=1
                correct += torch.eq(pred_q, target).sum().item() / len(img)
            print(cMatrix.value().shape)
            print("TOTAL", np.diagonal(cMatrix.value())[keep])




            results_mem_size[mem_size] = (max_acc, max_lr)
            logger.info("Final Max Result = %s", str(max_acc))
            writer.add_scalar('/finetune/best_' + str(lr_search), max_acc, args.limit)
        final_results_all.append((args.limit, results_mem_size))
        print("A=  ", results_mem_size)
        logger.info("Final Results = %s", str(results_mem_size))

        my_experiment.results["Final Results"] = final_results_all
        my_experiment.store_json()
        print("FINAL RESULTS = ", final_results_all)





    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--limit', type=int, default=600,
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--runs", type=int, default=50)

    args = argparser.parse_args()

    import os

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)
