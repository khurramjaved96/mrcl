import argparse

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import datasets.datasetfactory as df
import model.learner as learner
import model.modelfactory as mf
import utils
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

    dataset = df.DatasetFactory.get_dataset(args.dataset)

    if args.dataset == "CIFAR100":
        args.classes = list(range(50))

    if args.dataset == "omniglot":
        iterator = torch.utils.data.DataLoader(utils.remove_classes_omni(dataset, list(range(963))), batch_size=256,
                                               shuffle=True, num_workers=1)
    else:
        iterator = torch.utils.data.DataLoader(utils.remove_classes(dataset, args.classes), batch_size=256,
                                               shuffle=True, num_workers=1)

    logger.info(str(args))

    config = mf.ModelFactory.get_model("na", args.dataset)

    maml = learner.Learner(config).to(device)

    opt = torch.optim.Adam(maml.parameters(), lr=args.lr)

    for e in range(args.epoch):
        correct = 0
        for img, y in iterator:
            if e == 20:
                opt = torch.optim.Adam(maml.parameters(), lr=0.00001)
                logger.info("Changing LR from %f to %f", 0.0001, 0.00001)
            img = img.to(device)
            y = y.to(device)
            pred = maml(img)
            feature = F.relu(maml(img, feature=True))
            avg_feature = feature.mean(0)


            beta = args.beta
            beta_hat = avg_feature


            loss_rec = ((beta / (beta_hat+0.0001)) - torch.log(beta / (beta_hat+0.0001)) - 1)
            # loss_rec = (beta / (beta_hat)
            loss_rec = loss_rec * (beta_hat>beta).float()

            loss_sparse = loss_rec


            if args.l1:
                loss_sparse = feature.mean(0)
            loss_sparse = loss_sparse.mean()



            opt.zero_grad()
            loss = F.cross_entropy(pred, y)
            loss_sparse.backward(retain_graph=True)
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float()/ len(y)
        logger.info("Accuracy at epoch %d = %s", e, str(correct/len(iterator)))
        torch.save(maml, my_experiment.path + "model.net")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--beta', type=float, help='epoch number', default=0.1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--l1", action="store_true")
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.0001)
    argparser.add_argument('--classes', type=int, nargs='+', help='Total classes to use in training',
                           default=[0, 1, 2, 3, 4])
    argparser.add_argument('--name', help='Name of experiment', default="baseline")

    args = argparser.parse_args()

    args.name = "/".join([args.dataset, "baseline", str(args.epoch).replace(".", "_"), args.name])

    main(args)
