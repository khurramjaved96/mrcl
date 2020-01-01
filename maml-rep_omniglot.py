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


def main(args):
    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    args.classes = list(range(963))

    args.traj_classes = list(range(963))
    #
    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=False, train=True, all=True)


    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset)

    sampler_test = ts.SamplerFactory.get_sampler(args.dataset, list(range(600)), dataset_test, dataset_test)

    config = mf.ModelFactory.get_model("na", "omniglot-fc")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    utils.freeze_layers(args.rln, maml)

    for step in range(args.steps):

        t1 = np.random.choice(args.traj_classes, args.tasks, replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_few_shot_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, reset=not args.no_reset)
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
        #
        # Evaluation during training for sanity checks
        if step % 20 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            logger.info('step: %d \t training acc %s', step, str(accs))
            logger.info("Loss = %s", str(loss[-1].item()))
        if step % 600 == 599:
            torch.save(maml.net, my_experiment.path + "learner.model")
            accs_avg = None
            for temp_temp in range(0, 40):
                t1_test = np.random.choice(list(range(600)), args.tasks, replace=False)

                d_traj_test_iterators = []
                for t in t1_test:
                    d_traj_test_iterators.append(sampler_test.sample_task([t]))


                x_spt, y_spt, x_qry, y_qry = maml.sample_few_shot_training_data(d_traj_test_iterators, None,
                                                                                steps=args.update_step,
                                                                                reset=not args.no_reset)
                if torch.cuda.is_available():
                    x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                accs, loss = maml.finetune(x_spt, y_spt, x_qry, y_qry)
                if accs_avg is None:
                    accs_avg = accs
                else:
                    accs_avg += accs
            logger.info("Loss = %s", str(loss[-1].item()))
            writer.add_scalar('/metatest/train/accuracy', accs_avg[-1]/40, step)
            logger.info('TEST: step: %d \t testing acc %s', step, str(accs_avg/40))

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_fewshot")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset+"_fewshot", str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
