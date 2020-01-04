import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment

from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')


import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(256, 12)  # 6*6 from image dimension

        # self.fc2 = nn.Linear(200, 40)  # 6*6 from image dimension

    def forward(self, x):

        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x






def main(args):
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = torch.load(args.model, map_location='cpu').to(device)

    # config = mf.ModelFactory.get_model("na", "celeba")
    #
    # maml = MetaLearingClassification(args, config).to(device).net
    #
    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True, path=args.dataset_path, RLN=maml)

    iterator = torch.utils.data.DataLoader(dataset,
                                batch_size=128,
                                shuffle=True, num_workers=0)

    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True, path=args.dataset_path,
                                            RLN=maml)

    iterator_test = torch.utils.data.DataLoader(dataset_test,
                                           batch_size=128,
                                           shuffle=True, num_workers=0)

    #
    model = Linear().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info("GETS HERE")
    for step in range(args.steps):
        if step==150:
            logger.info("Changing LR to %s", str(args.lr*0.1))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*0.1)
        if step==300:
            logger.info("Changing LR to %s", str(args.lr*0.01))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.01)
        if step==450:
            logger.info("Changing LR to %s", str(args.lr*0.01))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.001)
        correct=None
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            # print(pred.shape, x.shape)
            loss = nn.BCEWithLogitsLoss()(pred, y.float())
            # logger.info(str(loss))
            loss.backward()
            optimizer.step()
            pred = torch.round(F.sigmoid(pred))
            # y = y.flatten()
            # print(pred==y)
            # print(pred.shape, y.shape)
            if correct is None:
                correct = torch.mean((pred.float() == y.float()).float(), 0)
            else:
                correct += torch.mean((pred.float() == y.float()).float(), 0)
            # print(correct)
        if step%20==0:
            logger.info("Correct Train percentage = %s", str(correct/len(iterator)))
            logger.info("Average Train = %s", str(torch.mean(correct/len(iterator))))
            correct = None
            for x, y in iterator_test:
                x = x.to(device)
                y = y.to(device)
                pred = F.sigmoid(model(x))
                pred = torch.round(pred)
                # y = y.flatten()
                if correct is None:
                    correct = torch.mean((pred.float() == y.float()).float(), 0)
                else:
                    correct += torch.mean((pred.float() == y.float()).float(), 0)
            logger.info("Correct Test percentage = %s", str(correct/len(iterator_test)))
            logger.info("Average Test = %s", str(torch.mean(correct / len(iterator))))
        # correct = None
        # for x, y in iterator:
        #     x, y = x.to(device), y.to(device)
        #
        #     pred = torch.round(F.sigmoid(model(x)).flatten())
        #     y = y.flatten()
        #
        #     correct = torch.sum(pred == y).float()/len(y)
        #     print("Correct percentage = ", correct)
        #     print(pred)
        #     quit()
            # y = y.flatten()

            #
            # print(pred)
            # quit()
            # loss = nn.BCELoss()(pred, y)
            # loss.backward()
            # optimizer.step()




#

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=400000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.0001)
    argparser.add_argument('--name', help='Name of experiment', default="celeba_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="celeba-linear")
    argparser.add_argument('--model', type=str, help='epoch number', default="/Volumes/Macintosh HD/Users/khurramjaved96/Beluga/MRCL_CELEB_FACES_LEN_50_0_003_0/learner.model")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset,"eval", str(args.lr).replace(".", "_"), args.name])
    print(args)
    main(args)
