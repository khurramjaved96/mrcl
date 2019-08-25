import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def clip_grad_params(self, params, norm=500):

        for p in params.parameters():
            g = p.grad
            # print(g)
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            # print(g)
            p.grad = g

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        counter = 0
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            assert(len(iterators)==1)
            rand_counter = 0
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                counter += 1
                if not counter % int(steps / len(iterators)) == 0:
                    x_traj.append(img)
                    y_traj.append(data)
                    # if counter % int(steps / len(iterators)) == 0:
                    #     class_cur += 1
                    #     break

                else:
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter==5:
                        break
            class_counter += 1


        # Sampling the random batch of data
        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)
        # print(y_traj)
        # print(y_rand)
        return x_traj, y_traj, x_rand, y_rand

    def sample_few_shot_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        counter = 0
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            # print("Itereator no ", class_counter)
            rand_counter = 0
            flag=True
            inner_counter=0
            for img, data in it1:

                class_to_reset = data[0].item()
                data[0] = class_counter
                # print(data[0])

                # if reset:
                #     # Resetting weights corresponding to classes in the inner updates; this prevents
                #     # the learner from memorizing the data (which would kill the gradients due to inner updates)
                #     self.reset_classifer(class_to_reset)

                counter += 1
                # print((counter % int(steps / len(iterators))) != 0)
                # print(counter)
                if inner_counter < 5:
                    x_traj.append(img)
                    y_traj.append(data)

                else:
                    flag = False
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter==5:
                        break
                inner_counter+=1
            class_counter += 1


        # Sampling the random batch of data
        # counter = 0
        # for img, data in it2:
        #     if counter == 1:
        #         break
        #     x_rand.append(img)
        #     y_rand.append(data)
        #     counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)


        x_rand = x_rand_temp
        y_rand = y_rand_temp

        x_traj, y_traj = torch.cat(x_traj).unsqueeze(0), torch.cat(y_traj).unsqueeze(0)

        # print(x_traj.shape, y_traj.shape)

        x_traj, y_traj = x_traj.expand(25, -1, -1, -1,-1)[0:5], y_traj.expand(25, -1)[0:5]

        # print(y_traj)
        # print(y_rand)
        # print(x_traj.shape, y_traj.shape, x_rand.shape, y_rand.shape)
        return x_traj, y_traj, x_rand, y_rand


    def inner_update(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()
        #
        grad = torch.autograd.grad(loss, fast_weights)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def clip_grad(self, grad, norm=50):
        grad_clipped = []
        for g, p in zip(grad, self.net.parameters()):
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """

        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        # print(y_traj)
        # print(y_rand)
        meta_losses = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(self.update_step + 1)]

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], True)

        with torch.no_grad():
            # Meta loss before any inner updates
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0], False)
            meta_losses[0] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy

            # Meta loss after a single inner update
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
            meta_losses[1] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        for k in range(1, self.update_step):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], True)

            # Computing meta-loss with respect to latest weights
            meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
            meta_losses[k + 1] += meta_loss

            # Computing accuracy on the meta and traj set for understanding the learning
            with torch.no_grad():
                pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy

        # Taking the meta gradient step
        self.optimizer.zero_grad()
        meta_loss = meta_losses[-1]
        meta_loss.backward()

        self.clip_grad_params(self.net, norm=5)

        self.optimizer.step()
        accuracies = np.array(accuracy_meta_set) / len(x_rand[0])

        return accuracies, meta_losses

    def finetune(self, x_traj, y_traj, x_rand, y_rand):
        """

        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        # print(y_traj)
        # print(y_rand)
        meta_losses = [0 for _ in range(10 + 1)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(10 + 1)]

        # Doing a single inner update to get updated weights

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)

        with torch.no_grad():
            # Meta loss before any inner updates
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0], False)
            meta_losses[0] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy

            # Meta loss after a single inner update
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
            meta_losses[1] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        for k in range(1, 10):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[0], fast_weights, y_traj[0], False)

            # Computing meta-loss with respect to latest weights
            meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
            meta_losses[k + 1] += meta_loss

            # Computing accuracy on the meta and traj set for understanding the learning
            with torch.no_grad():
                pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                # print("Accuracy at step", k, "= classification_accuracy")
                accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy


        accuracies = np.array(accuracy_meta_set) / len(x_rand[0])

        return accuracies, meta_losses


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.3)

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=True)
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            grad = torch.autograd.grad(loss, self.net.parameters())

            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            with torch.no_grad():

                logits = self.net(x_rand[0], vars=None, bn_training=True)

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q

            for k in range(1, len(x_traj)):
                logits = self.net(x_traj[k], fast_weights, bn_training=True)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits_q = self.net(x_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), :], fast_weights,
                                    bn_training=True)

                logits_select = []
                for no, val in enumerate(y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 1].long()):
                    logits_select.append(logits_q[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 0].unsqueeze(1))

                losses_q[k + 1] += loss_q

        self.optimizer.zero_grad()

        loss_q = losses_q[k + 1]
        loss_q.backward()

        self.optimizer.step()

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
