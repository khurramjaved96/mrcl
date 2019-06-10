import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    args.classes = list(range(963))

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    logger.info("Train set length = %d", len(iterator_train) * 5)
    logger.info("Test set length = %d", len(iterator_test) * 5)
    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    config = mf.ModelFactory.get_model("na", args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    for name, param in maml.named_parameters():
        param.learn = True
    for name, param in maml.net.named_parameters():
        param.learn = True

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("net.vars." + str(temp))

    for name, param in maml.named_parameters():
        # logger.info(name)
        if name in frozen_layers:
            logger.info("RLN layer %s", str(name))
            param.learn = False

    # Update the classifier
    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))

    for a in list_of_names:
        logger.info("TLN layer = %s", a[0])

    for step in range(args.steps):

        t1 = np.random.choice(args.classes, np.random.randint(1, args.tasks + 1), replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step)
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 40 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 300 == 0:
            correct = 0
            torch.save(maml.net, my_experiment.path + "learner.model")
            for img, target in iterator_test:
                with torch.no_grad():
                    img = img.to(device)
                    target = target.to(device)
                    logits_q = maml.net(img, vars=None, bn_training=False, feature=False)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct += torch.eq(pred_q, target).sum().item() / len(img)
            writer.add_scalar('/metatrain/test/classifier/accuracy', correct / len(iterator_test), step)
            logger.info("Test Accuracy = %s", str(correct / len(iterator_test)))
            correct = 0
            for img, target in iterator_train:
                with torch.no_grad():
                    img = img.to(device)
                    target = target.to(device)
                    logits_q = maml.net(img, vars=None, bn_training=False, feature=False)
                    pred_q = (logits_q).argmax(dim=1)
                    correct += torch.eq(pred_q, target).sum().item() / len(img)

            logger.info("Train Accuracy = %s", str(correct / len(iterator_train)))
            writer.add_scalar('/metatrain/train/classifier/accuracy', correct / len(iterator_train), step)


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
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
