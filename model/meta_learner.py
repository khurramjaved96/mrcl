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
        #
        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

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
    def inner_update(self, net, grad, adaptation_lr, list_of_context=None, log=False):
        counter = 0
        counter_lr = 0

        for (name, p) in net.named_parameters():
            if log:
                logger.debug("\nName = %s", name)
            if p.learn:
                if log:
                    logger.debug("Learn true")
                g = grad[counter]
                if self.context:
                    if log:
                        logger.debug("Context plasticity true")
                    if log:
                        logger.debug("Grad modified using context plasticity")
                    g = g * list_of_context[counter_lr].view(g.shape)
                    counter_lr += 1
                if self.plasticity:
                    if log:
                        logger.debug("Static plasticity true")
                    mask = net.meta_plasticity[counter]
                    if self.sigmoid:
                        if log:
                            logger.debug("Sigmoid true")
                        p.data -= adaptation_lr * g * torch.sigmoid(mask)
                    else:
                        p.data -= adaptation_lr * g * mask
                else:
                    p.data -= adaptation_lr * g
                counter += 1

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

    def meta_plasticity_mask(self):
        counter = 0
        return
        for (name, p) in self.net.named_parameters():
            if "meta" in name or "neuro" in name:
                pass
            else:
                if p.learn:

                    mask = self.net.meta_plasticity[counter]
                    if self.plasticity:
                        if self.sigmoid:
                            p.grad = (1 - torch.sigmoid(mask)) * p.grad

                    counter += 1

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        meta_losses = [0 for _ in range(len(x_traj) + 1)]

        prediction = self.net(x_traj[0], vars=None, bn_training=False)
        logits_select = []
        for no, val in enumerate(y_traj[0, :, 1].long()):
            logits_select.append(prediction[no, val])
        prediction = torch.stack(logits_select).unsqueeze(1)
        loss = F.mse_loss(prediction, y_traj[0, :, 0].unsqueeze(1))

        grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(self.net.parameters()))),
                                                  create_graph=True))

        if self.context:
            list_of_context = self.net.forward_plasticity(x_traj[0])

        fast_weights = []
        learn_counter = 0
        context_counter = 0

        for p in self.net.parameters():

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

        with torch.no_grad():
            prediction = self.net(x_rand[0], vars=None, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_rand[0, :, 1].long()):
                logits_select.append(prediction[no, val])
            prediction = torch.stack(logits_select).unsqueeze(1)
            loss_q = F.mse_loss(prediction, y_rand[0, :, 0].unsqueeze(1))
            meta_losses[0] += loss_q
        #
        for k in range(1, len(x_traj)):

            prediction = self.net(x_traj[k], fast_weights, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(prediction[no, val])
            prediction = torch.stack(logits_select).unsqueeze(1)

            loss = F.mse_loss(prediction, y_traj[k, :, 0].unsqueeze(1))

            grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, fast_weights)),
                                                      create_graph=True))

            if self.context:
                list_of_context = self.net.forward_plasticity(x_traj[k])

            fast_weights_new = []
            learn_counter = 0
            context_counter = 0
            #
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

            fast_weights = fast_weights_new

        logits_q = self.net(x_rand[0], fast_weights,
                            bn_training=True)
        #
        logits_select = []
        for no, val in enumerate(y_rand[0, :, 1].long()):
            logits_select.append(logits_q[no, val])
        prediction = torch.stack(logits_select).unsqueeze(1)
        # print(prediction.shape, y_rand.shape)
        loss_q = F.mse_loss(prediction, y_rand[0, :, 0].unsqueeze(1))

        meta_losses[k + 1] += loss_q

        if len(self.remaining_meta_weights) > 0:
            self.optimizer.zero_grad()
        if self.plasticity:
            self.optimizer_plastic.zero_grad()
        if self.neuro:
            self.optimizer_neuromodulation.zero_grad()
        if self.context:
            self.optimizer_context_models.zero_grad()

        # self.optimizer_plasticity.zero_grad()
        final_meta_loss = meta_losses[k + 1]
        final_meta_loss.backward()

        if self.plasticity:
            if self.sigmoid:
                self.meta_plasticity_mask()

        if len(self.remaining_meta_weights) > 0:
            self.optimizer.step()
        if self.plasticity:
            self.optimizer_plastic.step()
        if self.neuro:
            self.optimizer_neuromodulation.step()
        if self.attention:
            # logger.info("Updating attention")
            self.optimizer_attention.step()
        if self.context:
            # print("Changing context")
            self.optimizer_context_models.step()

        meta_losses[k + 1] = meta_losses[k + 1].detach()
        meta_losses[0] = meta_losses[0].detach()

        return meta_losses


# class MetaRL2(nn.Module):
#     """
#     Meta Learner
#     """
#
#     def __init__(self, args, config):
#         """
#         :param args:
#         """
#         super(MetaRL2, self).__init__()
#
#         self.update_lr = args["update_lr"]
#         self.meta_lr = args["meta_lr"]
#         self.update_step = args["update_step"]
#         self.plasticity = not args["no_plasticity"]
#         self.plasticity_lr = args["plasticity_lr"]
#         self.second_order = args['second_order']
#         self.sigmoid = not args["no_sigmoid"]
#         self.plasticity_decay = args['plasticity_decay']
#
#         if args['model_path'] is not None:
#             net_old = Learner.Learner(config)
#
#             logger.info("Loading model from path %s", args["model_path"])
#             self.net = torch.load(args['model_path'], map_location='cpu')
#             list_of_param = list(net_old.parameters())
#             for counter, old in enumerate(self.net.parameters()):
#                 old.learn = list_of_param[counter].learn
#             for (name, param) in self.net.named_parameters():
#                 if "meta" in name:
#                     param.learn = False
#                 # print("New", name, param.learn)
#         else:
#             self.net = Learner.Learner(config)
#         plastic_weights = []
#         other_weights = []
#         for name, param in self.net.named_parameters():
#             if "meta" in name:
#                 plastic_weights.append(param)
#             else:
#                 other_weights.append(param)
#
#         self.optimizer = optim.Adam(other_weights, lr=self.meta_lr)
#         self.optimizer_plastic = optim.Adam(plastic_weights, lr=self.plasticity_lr)
#
#     def clip_grad(self, grad, norm=10):
#         grad_clipped = []
#         for g, p in zip(grad, self.net.parameters()):
#             g = (g * (g < norm).float()) + ((g > norm).float()) * norm
#             g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
#             grad_clipped.append(g)
#         return grad_clipped
#
#     def clip_grad_inplace(self, net, norm=10):
#         for p in net.parameters():
#             g = p.grad
#             p.grad = (g * (g < norm).float()) + ((g > norm).float()) * norm
#             g = p.grad
#             p.grad = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
#         return net
#
#     def forward(self, x_traj, y_traj, a_traj, x_rand, y_rand, a_rand):
#         """
#         :param x_traj:   [b, setsz, c_, h, w]
#         :param y_traj:   [b, setsz]
#         :param x_rand:   [b, querysz, c_, h, w]
#         :param y_rand:   [b, querysz]
#         :return:
#         """
#
#         # print(y_spt, y_qry)
#         losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
#         corrects = [0 for _ in range(self.update_step + 1)]
#         # print(y_qry, y_spt)
#
#         # 1. run the i-th task and compute loss for k=0
#
#         logits = self.net(x_traj[0], vars=None, bn_training=False)
#         # logger.info("Logits shape = %s", str(logits.shape))
#         # logger.info("Actions shape = %s", str(a_traj[0].shape))
#         # logger.info("Logits = %s", str(logits))
#         # logger.info("Action = %s", str(a_traj[0]))
#         logits = torch.gather(logits, 1, a_traj[0].unsqueeze(-1)).squeeze(-1)
#         # logger.info("Selection logits = %s", str(logits))
#
#         loss = F.smooth_l1_loss(y_traj[0], logits)
#
#         grad = self.clip_grad(torch.autograd.grad(loss, list(filter(lambda x: x.learn, list(self.net.parameters()))),
#                                                   create_graph=True))
#
#         fast_weights = []
#         counter = 0
#
#         for (name, p) in self.net.named_parameters():
#             if "meta" in name or "neuro" in name:
#                 pass
#             else:
#                 if p.learn:
#                     g = grad[counter]
#                     mask = self.net.meta_plasticity[counter]
#                     if self.plasticity:
#                         if self.sigmoid:
#                             temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
#                         else:
#                             temp_weight = p - self.update_lr * g * mask
#                     else:
#                         temp_weight = p - self.update_lr * g
#                     counter += 1
#                 else:
#                     temp_weight = p
#                 fast_weights.append(temp_weight)
#
#         for params_old, params_new in zip(self.net.parameters(), fast_weights):
#             params_new.learn = params_old.learn
#
#         # this is the loss and accuracy before first update
#         with torch.no_grad():
#             # [setsz, nway]
#
#             logits_q = self.net(x_rand[0], self.net.parameters(), bn_training=False)
#             logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
#             # print(logits_q.shape, y_rand[0].shape)
#             # 0/0
#             loss_q = F.smooth_l1_loss(y_rand[0], logits_q)
#             losses_q[0] += loss_q
#             #
#             correct = losses_q[0]
#             corrects[0] = corrects[0] + correct
#
#         # this is the loss and accuracy after the first update
#         with torch.no_grad():
#             # [setsz, nway]
#             logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
#             logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
#             loss_q = F.smooth_l1_loss(y_traj[0], logits_q)
#             losses_q[1] += loss_q
#
#             corrects[1] += losses_q[1]
#         #
#         for k in range(1, self.update_step):
#             # 1. run the i-th task and compute loss for k=1~K-1
#             logits = self.net(x_traj[k], fast_weights, bn_training=False)
#             logits = torch.gather(logits, 1, a_traj[k].unsqueeze(-1)).squeeze(-1)
#             loss = F.smooth_l1_loss(y_traj[k], logits)
#             # 2. compute grad on theta_pi
#             grad = self.clip_grad(torch.autograd.grad(loss, fast_weights, create_graph=True))
#             # 3. theta_pi = theta_pi - train_lr * grad
#
#             fast_weights_new = []
#             counter = 0
#             for p in fast_weights:
#                 if p.learn:
#                     g = grad[counter]
#                     mask = self.net.meta_plasticity[counter]
#                     if self.plasticity:
#                         if self.sigmoid:
#                             temp_weight = p - self.update_lr * g * torch.sigmoid(mask)
#                         else:
#                             temp_weight = p - self.update_lr * g * mask
#
#                     else:
#                         temp_weight = p - self.update_lr * g
#
#                     counter += 1
#                 else:
#                     temp_weight = p
#                 fast_weights_new.append(temp_weight)
#             fast_weights = fast_weights_new
#             #
#             for params_old, params_new in zip(self.other_params, fast_weights):
#                 params_new.learn = params_old.learn
#
#         logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
#         logits_q = torch.gather(logits_q, 1, a_rand[0].unsqueeze(-1)).squeeze(-1)
#         # loss_q will be overwritten and just keep the loss_q on last update step.
#         loss_q = F.smooth_l1_loss(y_rand[0], logits_q)
#         losses_q[k + 1] += loss_q
#
#         with torch.no_grad():
#             corrects[k + 1] += losses_q[k + 1]
#
#         # end of all tasks
#         # sum over all losses on query set across all tasks
#         self.optimizer.zero_grad()
#         self.optimizer_plastic.zero_grad()
#
#         loss_q = losses_q[k + 1]
#         loss_q.backward()
#
#         self.optimizer.step()
#         self.optimizer_plastic.step()
#         return losses_q
#

def main():
    pass


if __name__ == '__main__':
    main()
