import argparse

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import datasets.datasetfactory as df
import configs.classification.pretraining_parser as params
import model.learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
import logging

logger = logging.getLogger('experiment')

def main():
    p = params.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    dataset = df.DatasetFactory.get_dataset(args['dataset'], background=True, train=True,path=args["path"], all=True)


    iterator = torch.utils.data.DataLoader(dataset, batch_size=256,
                                           shuffle=True, num_workers=0)

    logger.info(str(args))

    config = mf.ModelFactory.get_model("na", args["dataset"])

    maml = learner.Learner(config).to(device)

    for k, v in maml.named_parameters():
        print(k, v.requires_grad)

    opt = torch.optim.Adam(maml.parameters(), lr=args["lr"])

    for e in range(args["epoch"]):
        correct = 0
        for img, y in tqdm(iterator):
            img = img.to(device)
            y = y.to(device)
            pred = maml(img)

            opt.zero_grad()
            loss = F.cross_entropy(pred, y.long())
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float() / len(y)
        logger.info("Accuracy at epoch %d = %s", e, str(correct / len(iterator)))
        torch.save(maml, my_experiment.path + "model.net")


if __name__ == '__main__':

    main()
