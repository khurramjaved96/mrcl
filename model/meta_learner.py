import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.update_step = args["update_step"]
        self.plasticity = not args["no_plasticity"]
        self.plasticity_lr = args["plasticity_lr"]
        self.second_order = args['second_order']
        self.plasticity_decay = args['plasticity_decay']

        if args['model_path'] is not None:
            net_old = Learner.Learner(config)
            logger.info("Loading model from path %s", args["model_path"])
            self.net = torch.load(args['model_path'], map_location='cpu')
            list_of_param = list(net_old.parameters())
            for counter, old in enumerate(self.net.parameters()):
                old.learn = list_of_param[counter].learn
            for (name, param) in self.net.named_parameters():
                if "meta" in name:
                    param.learn = False
                # print("New", name, param.learn)
        else:
            self.net = Learner.Learner(config)
        plastic_weights = []
        other_weights = []
        for name, param in self.net.named_parameters():
            if "meta" in name:
                plastic_weights.append(param)
            else:
                other_weights.append(param)

        self.optimizer = optim.Adam(other_weights, lr=self.meta_lr)
        self.optimizer_plastic = optim.Adam(plastic_weights, lr=self.plasticity_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [2000, 4000, 6000], 0.3)
        self.meta_optim_plastic = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_plastic, [2000, 4000, 6000], 0.3)

    #
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

    def clip_plasticity(self):
        for (name, param) in self.net.named_parameters():
            if "meta" in name:
                param.data = param.data * (param.data < 1).float() + (param.data >= 1).float()
                param.data = param.data * (param.data > 0).float()

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        logits = self.net(x_traj[0], vars=None, bn_training=False)
        logits_select = []
        for no, val in enumerate(y_traj[0, :, 1].long()):
            logits_select.append(logits[no, val])
        logits = torch.stack(logits_select).unsqueeze(1)
        loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
        grad = self.clip_grad(torch.autograd.grad(loss, self.net.parameters(), create_graph=self.second_order))

        fast_weights = []
        counter = 0
        for g, (name, p) in zip(grad, self.net.named_parameters()):
            if p.learn:
                mask = self.net.meta_vars[counter]
                # if counter==0:
                #     print(torch.sigmoid(mask))
                if self.plasticity:
                    temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
                else:
                    temp_weight = p - self.update_lr * g
                # if counter==0:
                #     print(mask)
                counter += 1
            else:
                temp_weight = p
            fast_weights.append(temp_weight)

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        with torch.no_grad():
            logits = self.net(x_rand[0], vars=None, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_rand[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
            losses_q[0] += loss_q
        #
        for k in range(1, len(x_traj)):
            logits = self.net(x_traj[k], fast_weights, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)

            loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
            grad = self.clip_grad(torch.autograd.grad(loss, fast_weights, create_graph=self.second_order))

            fast_weights_new = []
            counter = 0
            for g, p in zip(grad, fast_weights):
                if p.learn:
                    mask = self.net.meta_vars[counter]
                    # if counter==0:
                    #     print(torch.sigmoid(mask))
                    if self.plasticity:
                        temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
                    else:
                        temp_weight = p - self.update_lr * g

                    counter += 1
                else:
                    temp_weight = p
                fast_weights_new.append(temp_weight)
            fast_weights = fast_weights_new

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
        if self.plasticity:
            self.optimizer_plastic.zero_grad()
        # self.optimizer_plasticity.zero_grad()
        loss_q = losses_q[k + 1]
        loss_q.backward()

        self.optimizer.step()
        if self.plasticity:
            self.optimizer_plastic.step()

        # self.net.decay_plasticity(self.plasticity_decay)
        return losses_q


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.update_step = args["update_step"]
        self.plasticity = not args["no_plasticity"]
        self.plasticity_lr = args["plasticity_lr"]
        self.second_order = args['second_order']

        if args['model_path'] is not None:
            net_old = Learner.Learner(config)
            logger.info("Loading model from path %s", args["model_path"])
            self.net = torch.load(args['model_path'], map_location='cpu')
            list_of_param = list(net_old.parameters())
            for counter, old in enumerate(self.net.parameters()):
                old.learn = list_of_param[counter].learn
            for (name, param) in self.net.named_parameters():
                if "meta" in name:
                    param.learn = False
                # print("New", name, param.learn)
        else:
            self.net = Learner.Learner(config)
        plastic_weights = []
        other_weights = []
        for name, param in self.net.named_parameters():
            if "meta" in name:
                plastic_weights.append(param)
            else:
                print("Name of other param = ", name)
                other_weights.append(param)

        self.optimizer = optim.Adam(other_weights, lr=self.meta_lr)
        #
        # if self.plasticity:
        self.optimizer_plastic = optim.Adam(plastic_weights, lr=self.plasticity_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [2000, 4000, 6000], 0.3)
        self.meta_optim_plastic = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_plastic, [2000, 4000, 6000], 0.3)

    #

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def reset_TLN(self):

        for name, param in self.net.named_parameters():
            # logger.info(name)
            if param.learn:
                # logger.info("Resetting param %s", str(param.name))
                w = nn.Parameter(torch.ones_like(param))
                # logger.info("W shape = %s", str(len(w.shape)))
                if len(w.shape) > 1:
                    torch.nn.init.kaiming_normal_(w)
                else:
                    w = nn.Parameter(torch.zeros_like(param))
                param.data = w
                param.learn = True

    def clip_grad_params(self, params, norm=500):

        for p in params.parameters():
            g = p.grad
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            p.grad = g

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        counter = 1
        #
        x_rand_temp = []
        y_rand_temp = []

        class_counter = 0
        for it1 in iterators:
            assert (len(iterators) == 1)
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

    def inner_update(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()
        #
        grad = torch.autograd.grad(loss, fast_weights, create_graph=True)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    #
    def inner_update_inplace(self, x, y, bn_training):

        logits = self.net(x, None, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)
        fast_weights = None
        if fast_weights is None:
            fast_weights = self.net.parameters()
        #
        grad = torch.autograd.grad(loss, fast_weights)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

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


class MetaLearnerRegression2(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(MetaLearnerRegression2, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.update_step = args["update_step"]

        self.net = Learner.PLN(11, 1, 200)
        self.meta_net = Learner.Plasticity(11, 1, 200)
        # self.optimizer = optim.Adam(list(self.net.parameters()) + list(self.meta_net.parameters()), lr=self.meta_lr)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.3)

    def inner_update(self, loss, weights):

        if weights is None:
            fast_weights = self.net.parameters()
        else:
            fast_weights = []
            for key in weights:
                fast_weights.append(weights[key])

        grads = torch.autograd.grad(loss, fast_weights,
                                    create_graph=True)

        params = OrderedDict()

        for (name, param), plasticity, grad in zip(self.net.meta_named_parameters(), self.meta_net.parameters(), grads):

            if param.learn:

                # params[name] =  param - self.update_lr * grad * torch.sigmoid(plasticity)
                params[name] = param - self.update_lr * grad
                params[name].learn = True
            else:
                # assert(False)
                params[name] = param
            # print("After", params[name].shape)
            # print("Grad", grad.shape)
            # print("Plasticity", plasticity.shape)
            # if first_update:
            #     print(torch.sigmoid(plasticity))

        return params

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):

            logits = self.net(x_traj[0])
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            fast_weights = self.inner_update(loss, None)

            with torch.no_grad():

                logits = self.net(x_rand[0])

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q
            # print("Len of x_traj = ", len(x_traj))
            for k in range(1, len(x_traj)):
                # print("Current k = ", k)
                logits = self.net(x_traj[k], params=fast_weights)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))

                fast_weights = self.inner_update(loss, fast_weights)

        logits_q = self.net(x_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), :], params=fast_weights,
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
        # for a in self.meta_net.parameters():
        #     print("meta grad")
        #     print(a.grad)
        # for a in self.net.parameters():
        #     print("Net grad")
        #     print(a.grad)
        self.optimizer.step()
        # 0/0
        # self.net = model_olds[-1]

        return losses_q


class MetaRL2(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """
        :param args:
        """
        super(MetaRL2, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.update_step = args["update_step"]
        self.plasticity = not args["no_plasticity"]
        self.plasticity_lr = args["plasticity_lr"]
        self.second_order = args['second_order']
        self.plasticity_decay = args['plasticity_decay']

        if args['model_path'] is not None:
            net_old = Learner.Learner(config)
            logger.info("Loading model from path %s", args["model_path"])
            self.net = torch.load(args['model_path'], map_location='cpu')
            list_of_param = list(net_old.parameters())
            for counter, old in enumerate(self.net.parameters()):
                old.learn = list_of_param[counter].learn
            for (name, param) in self.net.named_parameters():
                if "meta" in name:
                    param.learn = False
                # print("New", name, param.learn)
        else:
            self.net = Learner.Learner(config)
        plastic_weights = []
        other_weights = []
        for name, param in self.net.named_parameters():
            if "meta" in name:
                plastic_weights.append(param)
            else:
                other_weights.append(param)

        self.optimizer = optim.Adam(other_weights, lr=self.meta_lr)
        self.optimizer_plastic = optim.Adam(plastic_weights, lr=self.plasticity_lr)

    def forget(self):
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

    def forward(self, x_traj, y_traj, a_traj, x_rand, y_rand, a_rand):
        """
        :param x_traj:   [b, setsz, c_, h, w]
        :param y_traj:   [b, setsz]
        :param x_rand:   [b, querysz, c_, h, w]
        :param y_rand:   [b, querysz]
        :return:
        """

        # print(y_spt, y_qry)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        # print(y_qry, y_spt)

        # 1. run the i-th task and compute loss for k=0

        logits = self.net(x_traj[0], vars=None, bn_training=False)
        # logger.info("Logits shape = %s", str(logits.shape))
        # logger.info("Actions shape = %s", str(a_traj[0].shape))
        # logger.info("Logits = %s", str(logits))
        # logger.info("Action = %s", str(a_traj[0]))
        logits = torch.gather(logits, 1, a_traj[0].unsqueeze(-1)).squeeze(-1)
        # logger.info("Selection logits = %s", str(logits))

        loss = F.smooth_l1_loss(y_traj[0], logits)

        grad = self.clip_grad(torch.autograd.grad(loss, self.net.parameters(), create_graph=self.second_order))
        fast_weights = []
        counter = 0
        for g, (name, p) in zip(grad, self.net.named_parameters()):
            if p.learn:
                mask = self.net.meta_vars[counter]
                temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
                counter += 1
            else:
                temp_weight = p
            fast_weights.append(temp_weight)

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]

            logits_q = self.net(x_rand[0], self.net.parameters(), bn_training=False)
            logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
            # print(logits_q.shape, y_rand[0].shape)
            # 0/0
            loss_q = F.smooth_l1_loss(y_rand[0], logits_q)
            losses_q[0] += loss_q
            #
            correct = losses_q[0]
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
            logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
            loss_q = F.smooth_l1_loss(y_traj[0], logits_q)
            losses_q[1] += loss_q

            corrects[1] += losses_q[1]
        #
        for k in range(1, self.update_step):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.net(x_traj[k], fast_weights, bn_training=False)
            logits = torch.gather(logits, 1, a_traj[k].unsqueeze(-1)).squeeze(-1)
            loss = F.smooth_l1_loss(y_traj[k], logits)
            # 2. compute grad on theta_pi
            grad = self.clip_grad(torch.autograd.grad(loss, fast_weights, create_graph=self.second_order))
            # 3. theta_pi = theta_pi - train_lr * grad

            fast_weights_new = []
            counter = 0
            for g, p in zip(grad, fast_weights):
                if p.learn:
                    mask = self.net.meta_vars[counter]

                    temp_weight = p - self.update_lr * g * torch.sigmoid(mask)

                    counter += 1
                else:
                    temp_weight = p
                fast_weights_new.append(temp_weight)
            fast_weights = fast_weights_new

            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

        logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
        logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.smooth_l1_loss(y_rand[0], logits_q)
        losses_q[k + 1] += loss_q

        with torch.no_grad():
            corrects[k + 1] += losses_q[k + 1]

        # end of all tasks
        # sum over all losses on query set across all tasks
        self.optimizer.zero_grad()
        self.optimizer_plastic.zero_grad()

        loss_q = losses_q[k + 1]
        loss_q.backward()

        self.optimizer.step()
        self.optimizer_plastic.step()
        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
