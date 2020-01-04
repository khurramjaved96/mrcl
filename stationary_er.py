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
from collections import namedtuple
import random
from torch.nn import functional as F

logger = logging.getLogger('experiment')

transition = namedtuple('transition', 'input, label')

class_increment = 0

def update_pln(rep, network, x, y, optimizer, bn_training=True):
    with torch.no_grad():
        x = rep.net(x, feature=True)
    logits = network(x, None, bn_training=bn_training)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
    correct = torch.eq(pred_q, y).sum().item()
    return loss, correct


def q(classes):
    c1 = np.random.choice(classes)
    return c1

def q_transition(current_class, total_classes):
    # if np.random.binomial(size=1, n=1, p= 0.80)[0] == 0:
    #     action = int(np.random.normal(2, 10))
    #     return  (current_class + action)%total_classes
    # else:
    #     return current_class % total_classes

    # action = round(np.random.normal(0.2, 0.2))
    # action = round(np.random.normal(0.1, 0.30))
    # return (current_class + action) % total_classes

    # if np.random.binomial(size=1, n=1, p=0.80)[0] == 0:

    # task_size = 5
    # temp =  np.random.binomial(size=1, n=1, p=0.05)[0]
    #
    # val = (int(current_class / task_size) + temp) % int(total_classes / task_size)
    # return (val * 5 + random.randint(0, task_size)) % total_classes

    global class_increment
    if np.random.binomial(size=1, n=1, p=0.999)[0] == 0:
        class_increment = random.randint(0, 25)

    action = round(np.random.normal(0.03, 5.0))
    return (current_class + action)%total_classes

#
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_meta(self, batch_size, adapatation):
        left = self.sample(batch_size)
        right = self.sample_trajectory(adapatation)
        return left + right
    def sample_trajectory(self, batch_size):

        if self.location - batch_size < 0:
            left = self.buffer[(self.location-batch_size) % len(self.buffer): len(self.buffer)]
            right = self.buffer[0: batch_size - (len(self.buffer) - (self.location-batch_size) % len(self.buffer))]
            return left + right
        else:
            return self.buffer[ self.location-batch_size: self.location]

    def sample_trajectory_random(self, batch_size):
        start_index = random.randint(0, self.location)
        if start_index - batch_size < 0:
            left = self.buffer[(start_index-batch_size) % len(self.buffer): len(self.buffer)]
            right = self.buffer[0: batch_size - (len(self.buffer) - (start_index-batch_size) % len(self.buffer))]
            return left + right
        else:
            return self.buffer[ start_index-batch_size: start_index]

def eval_pln(network, x, y, bn_training=True):

    logits = network(x, None, bn_training=bn_training)
    loss = F.cross_entropy(logits, y)

    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
    correct = torch.eq(pred_q, y).sum().item()
    return loss, correct

def main(args):
    global class_increment

    utils.set_seed(args.seed)

    my_experiment = experiment(args.name, args, "../results/", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # Using first 963 classes of the omniglot as the meta-training set
    if args.test:
        args.classes = list(range(650))
    else:
        args.classes = list(range(900))

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=not args.test, train=True, all=True, path=args.dataset_path)


    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, None)

    config = mf.ModelFactory.get_model("na", args.dataset)

    config_pln = mf.ModelFactory.get_model("na", args.dataset+"-pln")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    maml = MetaLearingClassification(args, config).to(device)
    if args.model is not None:
        if args.old:
            maml.net = torch.load(args.model, map_location='cpu').to(device)
        else:
            maml = torch.load(args.model, map_location='cpu').to(device)

    pln  = MetaLearingClassification(args, config_pln).to(device).net

    utils.freeze_layers(args.rln, maml)

    buffer = replay_buffer(8000)

    current_class = q(args.classes)
    loss = None
    correct = None
    meta_loss = None
    pln_optimizer = optim.SGD(pln.parameters(), lr=args.update_lr)
    for step in range(args.steps):
        # print(current_class)

        d_traj_iterators = sampler.sample_task([current_class])

        x_step, y_step = maml.sample_current(d_traj_iterators)
        y_step = y_step + class_increment
        buffer.add(x_step, y_step)
        # # print(y_qry)
        # # print(y_spt)
        if torch.cuda.is_available():
            x_step, y_step = x_step.cuda(), y_step.cuda()


        # accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
        #
        # loss_cur = 0

        if args.smart:
            loss_cur, correct_cur = update_pln(maml, pln, x_step, y_step, pln_optimizer)
        else:
            loss_cur, correct_cur = eval_pln(maml.net, x_step, y_step)
        if loss is None:
            loss = loss_cur.item()
            correct = correct_cur
        else:
            correct = correct*0.99 + correct_cur *0.01
            loss = loss*0.99 + loss_cur.item()*0.01

        current_class = q_transition(current_class, len(args.classes))

        if step%100==0:
            logger.info("Average loss = %s %s %s", str(loss), "Step = ", str(step))
            logger.info("Average accuracy =  %s", str(correct))
            writer.add_scalar('/average_loss/accuracy', loss, step)
    #     Meta-Learning
        if step % 1000 == 0:
            torch.save(maml, my_experiment.path + "learner.model")

        if step > 50:
            support_set = buffer.sample(32)
            x_spt, y_spt = [x[0] for x in support_set], [x[1] for x in support_set]
            x_spt, y_spt = torch.cat(x_spt), torch.cat(y_spt)
            # print(x_spt.shape, y_spt.shape)
            if torch.cuda.is_available():
                x_spt, y_spt = x_spt.cuda(), y_spt.cuda()

            logits = maml.net(x_spt, None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            maml.optimizer.zero_grad()
            loss.backward()
            maml.optimizer.step()
            #
            if meta_loss is None:
                meta_loss = loss.item()
            else:
                meta_loss = meta_loss * 0.99 + loss.item() * 0.01

            if step % 100 == 0:
                logger.info("Meta loss = %s", str(meta_loss))





    utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)

#         Resetting TLN network



#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--freq', type=int, help='meta batch size, namely task num', default=20)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--model', type=str, help='epoch number', default=None)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--memorize", action="store_true")
    argparser.add_argument("--old", action="store_true")
    argparser.add_argument("--test", action="store_true")
    argparser.add_argument("--smart", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    args = argparser.parse_args()

    args.name = "/".join([args.dataset+"_new_protocol", str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
