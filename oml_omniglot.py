import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter

import configs.classification.reg_parser as reg_parser
import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_classification import MetaLearingClassification

logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print("Selected args", args)

    # Using first 963 classes of the omniglot as the meta-training set
    args["classes"] = list(range(963))

    args["traj_classes"] = args["classes"]
    # ls
    dataset_test = df.DatasetFactory.get_dataset(args["dataset"], background=True, train=False, all=True, path= args["path"])
    dataset = df.DatasetFactory.get_dataset(args["dataset"], background=True, train=True, all=True, path=args["path"])


    # Iterators used for evaluation
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args["dataset"], args["classes"], dataset, dataset_test)

    config = mf.ModelFactory.get_model("na", args["dataset"])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    for step in range(args["epoch"]):

        t1 = np.random.choice(args["traj_classes"], args["tasks"], replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args["update_step"], reset=False)
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
   main()
