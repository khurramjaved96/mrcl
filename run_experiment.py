import argparse
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from configs.default import default_argparse

import datasets.datasetfactory as df
from torch import optim
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

import torch.nn.init as init

logger = logging.getLogger('experiment')


def main():
    p = default_argparse.DefaultParser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=args["commit"],
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    args["classes"] = list(range(963))

    args["traj_classes"] = list(range(963))

    dataset = df.DatasetFactory.get_dataset(args["dataset"], background=True, train=True, all=True,
                                            path=args["dataset_path"])

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args["dataset"], args["classes"], dataset, None)

    config = mf.ModelFactory.get_model("na", args["dataset"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    utils.freeze_layers(maml)

    # maml.optimizer = optim.Adam(list(filter(lambda x: not x.learn, maml.net.parameters())), lr=maml.meta_lr)
    for step in range(args["steps"]):

        np.random.shuffle(args["classes"])
        for counter, current_class in enumerate(args["classes"][0:10]):
            t1 = [current_class]
            # t1 = np.random.choice(args.traj_classes, args.tasks, replace=False)

            d_traj_iterators = []
            for t in t1:
                d_traj_iterators.append(sampler.sample_task([t]))

            # print(args.classes[0:counter])
            d_rand_iterator = sampler.sample_task_no_cache(args["classes"][0:counter + 1])

            x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                                   steps=args["update_step"],
                                                                   reset=not args["no_reset"])
            # print("Support= ", y_spt)
            # print("Query= ", y_qry)

            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

            # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
            accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

            maml.update_TLN(x_spt, y_spt)

            if counter % 40 == 39:
                writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
                logger.info('step: %d \t training acc %s', step, str(accs))

        logger.info("Before reset")
        utils.log_accuracy(maml, my_experiment, d_rand_iterator, device, writer, step)

        #         Resetting TLN network

        # maml.reset_TLN()
        logger.info("After reset")
        utils.log_accuracy(maml, my_experiment, d_rand_iterator, device, writer, step)

        # maml.reset_layer()


#
if __name__ == '__main__':


    main()
#
