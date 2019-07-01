import argparse
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment

logger = logging.getLogger('experiment')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", args.commit)

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    total_clases = [900]

    keep = list(range(total_clases[0]))

    dataset = utils.remove_classes_omni(
        df.DatasetFactory.get_dataset("omniglot", train=True, path=args.data_path, all=True), keep)
    iterator_sorted = torch.utils.data.DataLoader(
        utils.iterator_sorter_omni(dataset, False, classes=total_clases),
        batch_size=128,
        shuffle=True, num_workers=2)

    iterator = iterator_sorted

    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = torch.load(args.model, map_location='cpu')

    maml = maml.to(device)

    reps = []
    counter = 0

    fig, axes = plt.subplots(9, 4)
    with torch.no_grad():
        for img, target in iterator:
            print(counter)

            img = img.to(device)
            target = target.to(device)
            # print(target)
            rep = maml(img, vars=None, bn_training=False, feature=True)
            rep = rep.view((-1, 32, 72)).detach().cpu().numpy()
            rep_instance = rep[0]
            if args.binary:
                rep_instance = (rep_instance > 0).astype(int)
            if args.max:
                rep = rep / np.max(rep)
            else:
                rep = (rep > 0).astype(int)
            if counter < 36:
                print("Adding plot")
                axes[int(counter / 4), counter % 4].imshow(rep_instance, cmap=args.color)
                axes[int(counter / 4), counter % 4].set_yticklabels([])
                axes[int(counter / 4), counter % 4].set_xticklabels([])
                axes[int(counter / 4), counter % 4].set_aspect('equal')

            counter += 1
            reps.append(rep)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    plt.savefig(my_experiment.path + "instance_" + str(counter) + ".pdf", format="pdf")
    plt.clf()

    rep = np.concatenate(reps)
    averge_activation = np.mean(rep, 0)
    plt.imshow(averge_activation, cmap=args.color)
    plt.colorbar()
    plt.clim(0, np.max(averge_activation))
    plt.savefig(my_experiment.path + "average_activation.pdf", format="pdf")
    plt.clf()
    instance_sparsity = np.mean((np.sum(np.sum(rep, 1), 1)) / (64 * 36))
    print("Instance sparisty = ", instance_sparsity)
    my_experiment.results["instance_sparisty"] = str(instance_sparsity)
    lifetime_sparsity = (np.sum(rep, 0) / len(rep)).flatten()
    mean_lifetime = np.mean(lifetime_sparsity)
    print("Lifetime sparsity = ", mean_lifetime)
    my_experiment.results["lifetime_sparisty"] = str(mean_lifetime)
    dead_neuros = float(np.sum((lifetime_sparsity == 0).astype(int))) / len(lifetime_sparsity)
    print("Dead neurons percentange = ", dead_neuros)
    my_experiment.results["dead_neuros"] = str(dead_neuros)
    plt.hist(lifetime_sparsity)

    plt.savefig(my_experiment.path + "histogram.pdf", format="pdf")
    my_experiment.store_json()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--model', type=str, help='epoch number')
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")

    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")

    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--binary", action="store_true")
    argparser.add_argument("--max", action="store_true")
    argparser.add_argument('--color', help='Name of experiment', default="YlGn")
    args = argparser.parse_args()

    import os

    args.data_path = "../data/omni"

    args.name = "/".join([args.dataset, "representation", str(args.epoch).replace(".", "_"), args.name + args.color])

    main(args)
