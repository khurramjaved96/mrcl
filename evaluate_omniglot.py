import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.classification.class_parser_eval as class_parser_eval
import datasets.datasetfactory as df
import model.learner as Learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment

logger = logging.getLogger('experiment')


def load_model(args, config):
    if args['model_path'] is not None:
        net_old = Learner.Learner(config)
        # logger.info("Loading model from path %s", args["model_path"])
        net = torch.load(args['model_path'],
                         map_location="cpu")

        for (n1, old_model), (n2, loaded_model) in zip(net_old.named_parameters(), net.named_parameters()):
            # print(n1, n2, old_model.adaptation, old_model.meta)
            loaded_model.adaptation = old_model.adaptation
            loaded_model.meta = old_model.meta

        net.reset_vars()
    else:
        net = Learner.Learner(config)
    return net


def main():
    p = class_parser_eval.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)

    final_results_all = []
    temp_result = []
    args['schedule'] = [int(x) for x in args['schedule'].split(":")]
    total_clases = args['schedule']
    print(args["schedule"])
    for tot_class in total_clases:
        print("Classes current step = ", tot_class)
        lr_list = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        lr_all = []
        for lr_search in range(0, 5):

            keep = np.random.choice(list(range(650)), tot_class, replace=False)

            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args['path']), keep)
            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)
            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=not args['test'], background=False, path=args['path']),
                keep)
            iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                   shuffle=False, num_workers=1)

            gpu_to_use = rank % args["gpus"]
            if torch.cuda.is_available():
                device = torch.device('cuda:' + str(gpu_to_use))
                logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
            else:
                device = torch.device('cpu')

            config = mf.ModelFactory.get_model("na", args['dataset'], output_dimension=1000)
            max_acc = -1000
            for lr in lr_list:

                print(lr)
                maml = load_model(args, config)
                maml = maml.to(device)

                opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)

                for _ in range(0, 1):
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
                    logits_q = maml(img)

                    pred_q = (logits_q).argmax(dim=1)

                    correct += torch.eq(pred_q, target).sum().item() / len(img)

                logger.info(str(correct / len(iterator)))
                if (correct / len(iterator) > max_acc):
                    max_acc = correct / len(iterator)
                    max_lr = lr

            lr_all.append(max_lr)
            logger.info("Final Max Result = %s", str(max_acc))
            results_mem_size = (max_acc, max_lr)
            temp_result.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Temp Results = %s", str(results_mem_size))

            my_experiment.results["Temp Results"] = temp_result
            my_experiment.store_json()
            print("LR RESULTS = ", temp_result)

        from scipy import stats
        best_lr = float(stats.mode(lr_all)[0][0])

        logger.info("BEST LR %s= ", str(best_lr))

        for aoo in range(0, args['runs']):

            keep = np.random.choice(list(range(650)), tot_class, replace=False)

            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=True, background=False), keep)
            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)
            dataset = utils.remove_classes_omni(
                df.DatasetFactory.get_dataset("omniglot", train=not args['test'], background=False), keep)
            iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                   shuffle=False, num_workers=1)



            for mem_size in [args['memory']]:
                max_acc = -10
                max_lr = -10

                lr = best_lr

                # for lr in [0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
                maml = load_model(args, config)
                maml = maml.to(device)

                opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)

                for _ in range(0, 1):
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
                results_mem_size  = (max_acc, max_lr)
                logger.info("Final Max Result = %s", str(max_acc))
            final_results_all.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Final results = %s", str(results_mem_size))

            my_experiment.results["Final Results"] = final_results_all
            my_experiment.store_json()
            print("FINAL RESULTS = ", final_results_all)


if __name__ == '__main__':
    main()
