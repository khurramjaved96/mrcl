import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

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
                loaded_model.learn = old_model.learn
                loaded_model.meta = old_model.meta
        else:
            self.net = Learner.Learner(config)

    #

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def clip_grad(self, grad, norm=10):
        grad_clipped = []
        for g, p in zip(grad, self.net.parameters()):
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped

    def clip_grad_inplace(self, net, norm=10):
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

        fast_weights = []
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
            fast_weights.append(temp_weight)

        return fast_weights

    #
    def inner_update_inplace(self, x, y, bn_training):

        fast_weights = self.net.parameters()

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)

        grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(fast_weights))),
                                                  create_graph=True))

        if self.context:
            list_of_context = self.net.forward_plasticity(x)

        fast_weights = []
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
            fast_weights.append(temp_weight)

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
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

        for k in range(1, self.update_step):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], bn_training=False)
            #
            # Computing meta-loss with respect to latest weights
            # meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], True)
            meta_losses[k + 1] += 0
            accuracy_meta_set[k + 1] = 0
            # Computing accuracy on the meta and traj set for understanding the learning

        # Computing meta-loss with respect to latest weights
        meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], bn_training=True)
        meta_losses[k + 1] += meta_loss

        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            # print("Predictions = ", pred_q)
            # print("Prediction = ", pred_q)
            # print("GT = ", y_rand[0])
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

        for k in range(0, self.update_step):
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

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        counter = 1
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            rand_counter = 0
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                # print(int(steps / len(iterators)))
                if not counter % (int(steps / len(iterators)) + 1) == 0:
                    x_traj.append(img)
                    y_traj.append(data)
                    counter += 1
                    # if counter % int(steps / len(iterators)) == 0:
                    #     class_cur += 1
                    #     break

                else:
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter == 5:
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
        return x_traj, y_traj, x_rand, y_rand

    def sample_current(self, iterator):

        for img, data in iterator:
            x_traj = img
            y_traj = data

            return x_traj, y_traj

    def sample_combined_data(self, iterators, it2, steps=2, reset=True):

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
            flag = True
            inner_counter = 0
            for img, data in it1:

                class_to_reset = data[0].item()
                data[0] = class_counter + 800
                # print(data[0])

                # if reset:
                #     # Resetting weights corresponding to classes in the inner updates; this prevents
                #     # the learner from memorizing the data (which would kill the gradients due to inner updates)
                #     print("Resetted class = ", class_to_reset)
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
                    if rand_counter == 2:
                        break
                inner_counter += 1
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

        x_rand, y_rand = torch.stack(x_rand), torch.stack(y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        x_traj, y_traj = torch.cat(x_traj).unsqueeze(0), torch.cat(y_traj).unsqueeze(0)

        # print(x_traj.shape, y_traj.shape)

        x_traj, y_traj = x_traj.expand(5, -1, -1, -1, -1), y_traj.expand(5, -1)

        # print(y_traj)
        # print(y_rand)
        # quit()
        # print(x_traj.shape, x_rand.shape)
        # quit()
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
            flag = True
            inner_counter = 0
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
                    if rand_counter == 5:
                        break
                inner_counter += 1
            class_counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)

        x_rand = x_rand_temp
        y_rand = y_rand_temp

        x_traj, y_traj = torch.cat(x_traj).unsqueeze(0), torch.cat(y_traj).unsqueeze(0)

        # print(x_traj.shape, y_traj.shape)

        x_traj, y_traj = x_traj.expand(5, -1, -1, -1, -1), y_traj.expand(5, -1)

        return x_traj, y_traj, x_rand, y_rand
