import logging

import numpy as np
import torch
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


def eval_iterator(iterator, device, maml):
    correct = 0
    for img, target in iterator:
        img = img.to(device)
        target = target.to(device)
        logits_q = maml(img)

        pred_q = (logits_q).argmax(dim=1)

        correct += torch.eq(pred_q, target).sum().item() / len(img)
    return correct / len(iterator)

def train_iterator(iterator_sorted, device, maml, opt):
    for img, y in iterator_sorted:
        img = img.to(device)
        y = y.to(device)

        pred = maml(img)
        opt.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        opt.step()

def main():
    p = class_parser_eval.Parser()
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)

    data_train = df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args['path'])
    data_test = df.DatasetFactory.get_dataset("omniglot", train=False, background=False, path=args['path'])
    final_results_train = []
    final_results_test = []
    lr_sweep_results = []

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    config = mf.ModelFactory.get_model("na", args['dataset'], output_dimension=1000)

    maml = load_model(args, config)
    maml = maml.to(device)

    args['schedule'] = [int(x) for x in args['schedule'].split(":")]
    no_of_classes_schedule = args['schedule']
    print(args["schedule"])
    for total_classes in no_of_classes_schedule:
        lr_sweep_range = [0.03, 0.01, 0.003,0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        lr_all = []
        for lr_search_runs in range(0, 5):

            classes_to_keep = np.random.choice(list(range(650)), total_classes, replace=False)

            dataset = utils.remove_classes_omni(data_train, classes_to_keep)

            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=no_of_classes_schedule),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)

            dataset = utils.remove_classes_omni(data_train, classes_to_keep)
            iterator_train = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                         shuffle=False, num_workers=1)

            max_acc = -1000
            for lr in lr_sweep_range:

                maml.reset_vars()

                opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)

                train_iterator(iterator_sorted, device, maml, opt)

                correct = eval_iterator(iterator_train, device, maml)
                if (correct > max_acc):
                    max_acc = correct
                    max_lr = lr

            lr_all.append(max_lr)
            results_mem_size = (max_acc, max_lr)
            lr_sweep_results.append((total_classes, results_mem_size))

            my_experiment.results["LR Search Results"] = lr_sweep_results
            my_experiment.store_json()
            logger.debug("LR RESULTS = %s", str(lr_sweep_results))

        from scipy import stats
        best_lr = float(stats.mode(lr_all)[0][0])

        logger.info("BEST LR %s= ", str(best_lr))

        for current_run in range(0, args['runs']):

            classes_to_keep = np.random.choice(list(range(650)), total_classes, replace=False)

            dataset = utils.remove_classes_omni(data_train, classes_to_keep)

            iterator_sorted = torch.utils.data.DataLoader(
                utils.iterator_sorter_omni(dataset, False, classes=no_of_classes_schedule),
                batch_size=1,
                shuffle=args['iid'], num_workers=2)

            dataset = utils.remove_classes_omni(data_test, classes_to_keep)
            iterator_test = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                        shuffle=False, num_workers=1)

            dataset = utils.remove_classes_omni(data_train, classes_to_keep)
            iterator_train = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                         shuffle=False, num_workers=1)

            lr = best_lr

            maml.reset_vars()

            opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)

            train_iterator(iterator_sorted, device,maml, opt)

            logger.info("Result after one epoch for LR = %f", lr)

            correct = eval_iterator(iterator_train, device, maml)

            correct_test = eval_iterator(iterator_test, device, maml)

            results_mem_size = (correct, best_lr, "train")
            logger.info("Final Max Result train = %s", str(correct))
            final_results_train.append((total_classes, results_mem_size))

            results_mem_size = (correct_test, best_lr, "test")
            logger.info("Final Max Result test= %s", str(correct_test))
            final_results_test.append((total_classes, results_mem_size))

            my_experiment.results["Final Results"] = final_results_train
            my_experiment.results["Final Results Test"] = final_results_test
            my_experiment.store_json()
            logger.debug("FINAL RESULTS = %s", str(final_results_train))
            logger.debug("FINAL RESULTS = %s", str(final_results_test))


if __name__ == '__main__':
    main()
