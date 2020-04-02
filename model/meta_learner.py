import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearnerRegression(nn.Module):

    def __init__(self, args, config, backbone_config):
        """
        #
        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.static_plasticity = args["static_plasticity"]
        self.context_plasticity = args["context_plasticity"]

        self.sigmoid = not args["no_sigmoid"]

        self.load_model(args, config, backbone_config)
        self.optimizers = []

        forward_meta_weights = self.net.get_forward_meta_parameters()
        if len(forward_meta_weights) > 0:
            self.optimizer_forward_meta = optim.Adam(forward_meta_weights, lr=self.meta_lr)
            self.optimizers.append(self.optimizer_forward_meta)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        if args["static_plasticity"]:
            self.net.add_static_plasticity()
            self.optimizer_static_plasticity = optim.Adam(self.net.static_plasticity, lr=args["plasticity_lr"])
            self.optimizers.append(self.optimizer_static_plasticity)

        self.optimizer_backbone = optim.Adam(self.net.context_backbone, lr=args["meta_lr"])
        self.optimizers.append(self.optimizer_backbone)

        if args["context_plasticity"]:
            self.net.add_context_plasticity(args['context_dimension'])
            self.optimizer_context_models = optim.Adam(self.net.context_plasticity_models, lr=args["context_lr"])
            self.optimizers.append(self.optimizer_context_models)

        if args["neuro"]:
            self.net.add_neuromodulation(args['context_dimension'])
            self.optimizer_neuro = optim.Adam(self.net.neuromodulation_parameters, lr=args["neuro_lr"])
            self.optimizers.append(self.optimizer_neuro)

        if args["meta_strength"]:
            self.net.add_meta_strength()
            # self.optimizer_meta_strength = optim.Adam(self.net.meta_strength, lr=args["strength_lr"])
            # self.optimizers.append(self.optimizer_meta_strength)

        self.log_model()

    def log_model(self):
        for name, param in self.net.named_parameters():
            # print(name)
            if param.meta:
                logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
            if param.adaptation:
                logger.debug("Weight for adaptation = %s %s", name, str(param.shape))

    def optimizer_zero_grad(self):
        for opti in self.optimizers:
            opti.zero_grad()

    def optimizer_step(self):
        for opti in self.optimizers:
            opti.step()

    def load_model(self, args, config, context_config):
        if args['model_path'] is not None:
            net_old = Learner.Learner(config, context_config)
            self.net = torch.load(args['model_path'] + "/net.model",
                                  map_location="cpu")

            for (n1, old_model), (n2, loaded_model) in zip(net_old.named_parameters(), self.net.named_parameters()):
                loaded_model.adaptation = old_model.adaptation
                loaded_model.meta = old_model.meta
        else:
            self.net = Learner.Learner(config, context_config)

    def inner_update(self, net, vars, grad, adaptation_lr, list_of_context=None, log=False):
        adaptation_weight_counter = 0

        new_weights = []
        for p in vars:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                if self.context_plasticity:
                    g = g * list_of_context[adaptation_weight_counter].view(g.shape)
                if self.static_plasticity:
                    mask = net.static_plasticity[adaptation_weight_counter].view(g.shape)
                    g = g * torch.sigmoid(mask)

                temp_weight = p - adaptation_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

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

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        prediction = self.net(x_traj[0], vars=None)
        loss = F.mse_loss(prediction, y_traj[0, :, 0].unsqueeze(1))

        grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(),
                                                  create_graph=True))

        list_of_context = None
        if self.context_plasticity:
            list_of_context = self.net.forward_plasticity(x_traj[0])

        fast_weights = self.inner_update(self.net, self.net.parameters(), grad, self.update_lr, list_of_context)

        with torch.no_grad():
            prediction = self.net(x_rand[0], vars=None)
            first_loss = F.mse_loss(prediction, y_rand[0, :, 0].unsqueeze(1))

        for k in range(1, len(x_traj)):

            prediction = self.net(x_traj[k], fast_weights)

            loss = F.mse_loss(prediction, y_traj[k, :, 0].unsqueeze(1))

            grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights),
                                                      create_graph=True))

            list_of_context = None
            if self.context_plasticity:
                list_of_context = self.net.forward_plasticity(x_traj[k])

            fast_weights = self.inner_update(self.net, fast_weights, grad, self.update_lr, list_of_context)

        prediction_qry_set = self.net(x_rand[0], fast_weights)
        # print(prediction_qry_set)
        final_meta_loss = F.mse_loss(prediction_qry_set, y_rand[0, :, 0].unsqueeze(1))

        self.optimizer_zero_grad()

        final_meta_loss.backward()

        self.optimizer_step()

        return [first_loss.detach(), final_meta_loss.detach()]


def main():
    pass


if __name__ == '__main__':
    main()
