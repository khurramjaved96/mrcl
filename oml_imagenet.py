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
import datasets.miniimagenet as imgnet

logger = logging.getLogger('experiment')


def main(args):
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    args.classes = list(range(64))

    # args.traj_classes = list(range(int(64 / 2), 963))

    dataset = imgnet.MiniImagenet(args.dataset_path, mode='train')

    dataset_test = imgnet.MiniImagenet(args.dataset_path, mode='test')

    # Iterators used for evaluation
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset)

    config = mf.ModelFactory.get_model("na", "imagenet")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    utils.freeze_layers(args.rln, maml)

    for step in range(args.steps):

        t1 = np.random.choice(args.classes, args.tasks, replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, reset=not args.no_reset)
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

        # Evaluation during training for sanity checks
        if step % 40 == 39:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 300 == 299:
            utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)
            utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)


#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=200000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="imagenet")
    argparser.add_argument('--dataset-path', help='Dataset path', default="~/")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
