import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter

import datasets.datasetfactory as df
from torch import optim
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')


def main(args):
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    args.classes = list(range(963))

    args.traj_classes = list(range(963))

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True, path=args.dataset_path)


    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, None)

    config = mf.ModelFactory.get_model("na", args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    utils.freeze_layers(args.rln, maml)

    maml.optimizer = optim.Adam(list(filter(lambda x: not x.learn, maml.net.parameters())), lr=maml.meta_lr)

    for step in range(args.steps):

        np.random.shuffle(args.classes)
        for counter, current_class in enumerate(args.classes):
            t1 = [current_class]
            # t1 = np.random.choice(args.traj_classes, args.tasks, replace=False)

            d_traj_iterators = []
            for t in t1:
                d_traj_iterators.append(sampler.sample_task([t]))

            # print(args.classes[0:counter])
            d_rand_iterator = sampler.sample_task_no_cache(args.classes[0:counter+1])

            x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                                   steps=args.update_step, reset=not args.no_reset)
            # print(y_qry)
            # print(y_spt)
            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

            maml.update_TLN(x_spt, y_spt)



            if counter % 40 == 39:
                writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
                logger.info('step: %d \t training acc %s', step, str(accs))

        utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)

#         Resetting TLN network

        maml.reset_TLN()
        utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)

        maml.reset_layer()

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset+"_new_protocol", str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
