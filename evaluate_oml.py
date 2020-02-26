import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.classification.classification_parser_eval as reg_parser
import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
from model.meta_classification import MetaLearingClassification

logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", False)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    print(logger.info(args["dataset"]))
    config = mf.ModelFactory.get_model("na", args["dataset"])

    final_results_all = []
    temp_result = []
    total_clases = [args["schedule"]]
    for tot_class in total_clases:
        lr_list = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001, 0.0000003, 0.0000001]
        lr_all = []
        for lr_search in range(0, 5):

            keep = np.random.choice(list(range(650)), tot_class, replace=False)
            # np.random.shuffle(keep)

            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=True, background=True, path=args["path"]), keep)
            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                batch_size=1,
                shuffle=args["iid"], num_workers=2)
            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=not args["test"], background=True,
                                              path=args["path"]),
                keep)
            iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                   shuffle=False, num_workers=1)

            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}
            print("Gets here")
            for mem_size in [args["memory"]]:
                max_acc = -10
                max_lr = -10
                for lr in lr_list:


                    # for lr in [0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
                    maml = MetaLearingClassification(args, config).to(device).net

                    maml = maml.to(device)


                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))

                    opt = torch.optim.Adam(list_of_params, lr=lr)

                    for _ in range(0, args["epoch"]):
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
                writer.add_scalar('/finetune/best_' + str(lr_search), max_acc, tot_class)
            temp_result.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Temp Results = %s", str(results_mem_size))

            my_experiment.results["Temp Results"] = temp_result
            my_experiment.store_json()
            print("LR RESULTS = ", temp_result)

        from scipy import stats
        best_lr = float(stats.mode(lr_all)[0][0])

        logger.info("BEST LR %s= ", str(best_lr))

        for aoo in range(0, args["runs"]):

            keep = np.random.choice(list(range(650)), tot_class, replace=False)
            #
            if args["dataset"] == "omniglot":

                dataset = utils.remove_classes_omni(
                    df.DatasetFactory.get_dataset("omniglot", train=True, background=False), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                    batch_size=1,
                    shuffle=args["iid"], num_workers=2)
                dataset = utils.remove_classes_omni(
                    df.DatasetFactory.get_dataset("omniglot", train=not args["test"], background=False), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                       shuffle=False, num_workers=1)
            elif args["dataset"] == "CIFAR100":
                keep = np.random.choice(list(range(50, 100)), tot_class)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args["dataset"], train=True), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter(dataset, False, classes=tot_class),
                    batch_size=16,
                    shuffle=args["iid"], num_workers=2)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args["dataset"], train=False), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                       shuffle=False, num_workers=1)
            # sampler = ts.MNISTSampler(list(range(0, total_clases)), dataset)
            #
            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args["memory"]]:
                max_acc = -10
                max_lr = -10

                lr = best_lr

                # for lr in [0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
                maml = torch.load(args["model"], map_location='cpu')

                if args["scratch"]:
                    config = mf.ModelFactory.get_model("na", args["dataset"])
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
                        if args["reset"]:
                            w = nn.Parameter(torch.ones_like(param))
                            # logger.info("W shape = %s", str(len(w.shape)))
                            if len(w.shape) > 1:
                                torch.nn.init.kaiming_normal_(w)
                            else:
                                w = nn.Parameter(torch.zeros_like(param))
                            param.data = w
                            param.learn = True

                frozen_layers = []
                for temp in range(args["rln"] * 2):
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
                if args["scratch"] or args["no_freeze"]:
                    print("Empty filter list")
                    list_of_params = maml.parameters()
                #
                for x in list_of_names:
                    logger.info("Unfrozen layer = %s", str(x[0]))
                opt = torch.optim.Adam(list_of_params, lr=lr)

                for _ in range(0, args["epoch"]):
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
    main()
