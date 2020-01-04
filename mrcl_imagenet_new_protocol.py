import argparse
import logging
from random import sample

import numpy as np
import torch
import torch.utils.model_zoo
from torch.utils.tensorboard import SummaryWriter

import datasets.datasetfactory as df
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
from mpi4py import MPI

logger = logging.getLogger('experiment')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main(args):
    torch.manual_seed(rank)
    utils.set_seed(rank)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    my_experiment = experiment(args.name + "_" + rank, args, args.output, commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True, path=args.dataset_path)

    # print("Sorting")
    # indices = np.sort(sample(list(range(len(dataset.samples))), 40000))
    # dataset.samples = [dataset.samples[a] for a in indices]
    # print("Sorted and subsampled")
    train_iterator = torch.utils.data.DataLoader(dataset,
                                                 batch_size=40,
                                                 shuffle=True, num_workers=2)

    config = mf.ModelFactory.get_model("na", "fullimagenet")

    maml = MetaLearingClassification(args, config).to(device)

    if args.resume is not None:
        maml = torch.load(args.resume)
    utils.freeze_layers(args.rln, maml)
    # maml = maml.to(device)
    classes_per_list = 20
    inner = 0
    inner_eval = 0
    print("Length of train iterator = ", len(train_iterator))
    for step in range(args.steps):
        for traj, y in train_iterator:

            # Switching labels (So tasks are mutually exclusive)
            indices_temp = np.unique(y.data.numpy())
            map = {}
            y_shape = y.shape
            y = y.flatten()
            for val, a in enumerate(indices_temp):
                map[a] = val
                y[(y == a).nonzero()] = val

            # print(traj.shape)
            traj = traj.view(-1, 3, 224, 224)

            for bptt in range(0, len(traj), 10):
                inner += 1
                x_spt = traj[bptt:bptt + 5].unsqueeze(1)
                y_spt = y[bptt:bptt + 5].unsqueeze(1)

                # print(x_spt.shape)
                # print(y_spt)

                # x_spt = x_spt.reshape(5, 8, 3, 224, 224)
                # y_spt = y_spt.reshape(5, 8)

                indices = []
                if not bptt == 0:
                    indices = sample(list(range(bptt)), min(bptt + 10, 10))
                indices = indices + list(range(bptt + 5, bptt + 10))
                # print(indices)
                # indices = indices +
                # print(indices)
                x_qry = traj[indices].unsqueeze(0)
                y_qry = y[indices].unsqueeze(0)
                # print("Support", y_spt.flatten())
                # print("Query", y_qry.flatten())
                if torch.cuda.is_available():
                    x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                # print("x_spt, y_spt, x_qry, y_qry", x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)
                # print("Y Support", y_spt.flatten())
                # print("Y query", y_qry.flatten())
                accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
                maml.update_TLN(x_spt, y_spt)
                writer.add_scalar('/metatrain/train/accuracy', accs[-1], inner)

                # print(spt.shape)
            logger.info('step: %d \t training acc %s', inner, str(accs))
            # maml.reset_layer()
            if args.store and inner % 1000 == 0:
                torch.save(maml, my_experiment.path + "learner.model")
            if inner % 500 == 40:
                accuracies = []
                for traj, y in train_iterator:
                    inner_eval += 1
                    # Switching labels (So tasks are mutually exclusive)
                    indices_temp = np.unique(y.data.numpy())
                    map = {}
                    y_shape = y.shape
                    y = y.flatten()
                    for val, a in enumerate(indices_temp):
                        map[a] = val
                        y[(y == a).nonzero()] = val

                    # print(traj.shape)
                    traj = traj.view(-1, 3, 224, 224)

                    for bptt in range(0, len(traj), 10):

                        x_spt = traj[bptt:bptt + 10].unsqueeze(1)
                        y_spt = y[bptt:bptt + 10].unsqueeze(1)

                        indices = []
                        if not bptt == 0:
                            indices = sample(list(range(bptt)), min(bptt + 10, 10))
                        indices = indices + list(range(bptt + 5, bptt + 10))

                        x_qry = traj[indices].unsqueeze(0)
                        y_qry = y[indices].unsqueeze(0)

                        if torch.cuda.is_available():
                            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(
                                device)

                        maml.update_TLN(x_spt, y_spt)

                    indices = sample(list(range(len(traj))), 128)

                    traj, y = traj[indices], y[indices]
                    traj, y = traj.to(device), y.to(device)
                    meta_loss, last_layer_logits = maml.meta_loss(traj, None, y, False, grad=False)

                    classification_accuracy = maml.eval_accuracy(last_layer_logits, y)
                    accuracies.append(classification_accuracy / 128)

                    if inner_eval % 10 == 0:
                        break
                    writer.add_scalar('/metatrain/val/accuracy', classification_accuracy / 128, inner_eval)
                logger.info("Incremental classifier avg accuracy = %s", np.mean(accuracies))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--gpu', type=int, help='epoch number', default=0)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="imagenet")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--store", action="store_true")
    argparser.add_argument("--no-pretrained", action="store_true")
    argparser.add_argument('--output', help='Name of experiment', default="/localscratch/khurram.3032680.0/result")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--resume', help='Name of experiment', default=None)
    argparser.add_argument("--rln", type=int, default=8)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset + "_new_protocol", str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
#
