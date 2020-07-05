import logging
import random

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

        self.update_lr = args["update_lr"]
        self.meta_model_parameters_lr = args["meta_lr"]
        self.update_step = args["update_step"]
        self.plasticity = not args["no_plasticity"]
        self.neuro = args["neuro"]
        self.sigmoid = not args["no_sigmoid"]
        self.remaining_meta_weights = []
        self.context = args["context_plasticity"]

        self.load_model(args, config)

        self.attention = False
        neuro_weights = []
        context_models = []
        plastic_weights = []
        other_meta_weights = []
        attention_weights = []
        for name, param in self.net.named_parameters():
            if "meta_plasticity" in name:
                logger.info("Meta plasticity weight = %s, %s", name, str(param.shape))
                plastic_weights.append(param)
            elif "neuromodulation" in name:
                logger.info("Neuro weight = %s, %s", name, str(param.shape))
                neuro_weights.append(param)
            elif "context_models" in name or "plasticity_backbone" in name:
                logger.info("Context plasticity weight = %s, %s", name, str(param.shape))
                context_models.append(param)
            elif "attention" in name:
                logger.info("Attention weight = %s, %s", name, str(param.shape))
                attention_weights.append(param)

            else:
                if param.meta:
                    logger.info("Other meta weights = %s, %s", name, str(param.shape))
                    other_meta_weights.append(param)

        logger.warning("PRINTING MORE TO BE SAFE")
        for name, param in self.net.named_parameters():
            if param.meta:
                logger.info("Weight in meta-optimizer = %s", name)
            if param.learn:
                logger.debug("Weight for adaptation = %s", name)

        self.remaining_meta_weights = other_meta_weights

        if len(self.remaining_meta_weights) > 0:
            self.optimizer = optim.Adam(other_meta_weights, lr=self.meta_model_parameters_lr)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        if len(plastic_weights) > 0:
            self.optimizer_plastic = optim.Adam(plastic_weights, lr=args["plasticity_lr"])
        if len(neuro_weights) > 0:
            self.optimizer_neuromodulation = optim.Adam(neuro_weights, lr=args["modulation_lr"])
        if len(attention_weights) > 0:
            self.optimizer_attention = optim.Adam(attention_weights, lr=args["attention_lr"])
            self.attention = True
        if self.context:
            self.optimizer_context_models = optim.Adam(context_models, lr=args["context_lr"])

    def load_model(self, args, config):
        if args['model_path'] is not None:
            net_old = Learner.Learner(config)
            # logger.info("Loading model from path %s", args["model_path"])
            self.net = torch.load(args['model_path'] + "/net.model",
                                  map_location="cpu")

            for (n1, old_model), (n2, loaded_model) in zip(net_old.named_parameters(), self.net.named_parameters()):
                print(n1, n2, old_model.learn, old_model.meta)
                loaded_model.learn = old_model.learn
                loaded_model.meta = old_model.meta
        else:
            self.net = Learner.Learner(config)

    def clip_grad(self, grad, norm=50):
        grad_clipped = []
        for g, p in zip(grad, self.net.parameters()):
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped

    def clip_grad_inplace(self, net, norm=50):
        for p in net.parameters():
            g = p.grad
            p.grad = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = p.grad
            p.grad = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
        return net

    def inner_update(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(fast_weights))),
                                                  create_graph=True))

        if self.context:
            list_of_context = self.net.forward_plasticity(x)
        # q
        new_fast_weights = []
        learn_counter = 0
        context_counter = 0

        for p in fast_weights:

            if p.learn:
                g = grad[learn_counter]
                if self.context:
                    g = g * list_of_context[context_counter].view(g.shape)
                    context_counter += 1
                if self.plasticity:
                    mask = self.net.meta_plasticity[learn_counter]
                    if self.sigmoid:
                        temp_weight = p - self.update_lr * g * torch.sigmoid(mask.view(g.shape))
                    else:
                        temp_weight = p - self.update_lr * g * mask.view(g.shape)
                else:

                    temp_weight = p - self.update_lr * g
                learn_counter += 1
                temp_weight.learn = True
            else:
                temp_weight = p
                temp_weight.learn = False
            new_fast_weights.append(temp_weight)

        return new_fast_weights

    #
    def inner_update_inplace(self, x, y, bn_training):

        fast_weights = self.net.parameters()

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)

        grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(fast_weights))),
                                                  create_graph=True))

        if self.context:
            list_of_context = self.net.forward_plasticity(x)

        fast_weights_new = []
        learn_counter = 0
        context_counter = 0

        for p in fast_weights:

            if p.learn:
                g = grad[learn_counter]
                if self.context:
                    g = g * list_of_context[context_counter].view(g.shape)
                    context_counter += 1
                if self.plasticity:
                    mask = self.net.meta_plasticity[learn_counter]
                    if self.sigmoid:
                        temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
                    else:
                        temp_weight = p - self.update_lr * g * mask
                else:

                    temp_weight = p - self.update_lr * g
                learn_counter += 1
                temp_weight.learn = True
            else:
                temp_weight = p
                temp_weight.learn = False
            fast_weights_new.append(temp_weight)

        for params_old, params_new in zip(self.net.parameters(), fast_weights_new):
            params_old.data = params_new.data

        return loss

    def meta_loss(self, x, fast_weights, y, bn_training, grad=True):
        if grad:
            logits = self.net(x, fast_weights, bn_training=bn_training)
            loss_q = F.cross_entropy(logits, y)
            return loss_q, logits
        else:
            with torch.no_grad():
                logits = self.net(x, fast_weights, bn_training=bn_training)
                loss_q = F.cross_entropy(logits, y)
                return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        # print("PRed = ", pred_q)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def eval_accuracy_images(self, x, y):
        acc = 0
        for x_cur, y_cur in zip(x, y):
            meta_loss, last_layer_logits = self.meta_loss(x_cur, self.net.parameters(), y_cur,
                                                          bn_training=False)

            classification_accuracy = self.eval_accuracy(last_layer_logits, y)
            acc += classification_accuracy
        return acc / len(x)

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
        meta_losses = [0 for _ in range(len(x_traj) + 1)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(len(x_traj) + 1)]

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], bn_training=False)

        with torch.no_grad():
            # Meta loss before any inner updates
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0],
                                                          bn_training=False)
            meta_losses[0] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy

            # Meta loss after a single inner update
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], bn_training=False)
            meta_losses[1] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        for k in range(1, len(x_traj)):
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], bn_training=False)

        meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], bn_training=True)
        meta_losses[k + 1] += meta_loss

        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
            accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy

        # Taking the meta gradient step
        self.optimizer.zero_grad()
        meta_loss = meta_losses[-1]
        meta_loss.backward()

        # self.clip_grad_params(self.net, norm=5)

        self.optimizer.step()

        accuracies = np.array(accuracy_meta_set) / len(x_rand[0])
        meta_losses = np.array(meta_losses)

        return accuracies, meta_losses

    def update_TLN(self, x_traj, y_traj):

        for k in range(0, len(x_traj)):
            self.inner_update_inplace(x_traj[k], y_traj[k], True)

        return None

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

    def sample_training_data(self, ids, labels, data, elem_per_class_train, limit=128):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []
        for counter, a in enumerate(ids):
            rand_index = random.randint(0, 20 - elem_per_class_train)
            start_train = a * 20 + rand_index
            end_train = start_train + elem_per_class_train

            images = data[start_train: end_train]

            assert (end_train - (a * 20) <= 20)
            assert (end_train - (a * 20) > 0)

            for img in images:
                x_traj.append(img)
                y_traj.append(labels[counter])

        x_traj, y_traj = torch.stack(x_traj), torch.tensor(y_traj)
        if len(x_traj) > limit:
            indices = np.random.choice(list(range(len(x_traj))), limit, replace=False)
            x_traj = x_traj[indices]
            y_traj = y_traj[indices]

        return x_traj, y_traj
