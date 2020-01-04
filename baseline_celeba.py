import argparse

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
import datasets.task_sampler as ts
from experiment.experiment import experiment
import logging

logger = logging.getLogger('experiment')

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    np.random.seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/")

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True, path=args.dataset_path)

    valid_classes = dataset.get_valid_classes()

    args.classes = valid_classes

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset)

    iterator  = sampler.sample_task_no_cache(valid_classes)

    logger.info(str(args))

    config = mf.ModelFactory.get_model("na", args.dataset)

    maml = learner.Learner(config).to(device)

    opt = torch.optim.Adam(maml.parameters(), lr=args.lr)

    for e in range(args.epoch):
        correct = 0
        for img, y in tqdm(iterator):
            if e == 50:
                opt = torch.optim.Adam(maml.parameters(), lr=0.00001)
                logger.info("Changing LR from %f to %f", 0.0001, 0.00001)
            img = img.to(device)
            y = y.to(device)
            pred = maml(img)
            feature = maml(img, feature=True)
            loss_rep = torch.abs(feature).sum()

            opt.zero_grad()
            loss = F.cross_entropy(pred, y)
            # loss_rep.backward(retain_graph=True)
            # logger.info("L1 norm = %s", str(loss_rep.item()))
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float()/ len(y)
        logger.info("Accuracy at epoch %d = %s", e, str(correct/len(iterator)))


        torch.save(maml, my_experiment.path + "model.net")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--dataset', help='Name of experiment', default="celeba")
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.0001)
    argparser.add_argument('--classes', type=int, nargs='+', help='Total classes to use in training',
                           default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--name', help='Name of experiment', default="baseline")

    args = argparser.parse_args()

    args.name = "/".join([args.dataset, "baseline", str(args.epoch).replace(".", "_"), args.name])

    main(args)
