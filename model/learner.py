import logging

logger = logging.getLogger("experiment")

import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import oml

logger = logging.getLogger("experiment")


class Learner(nn.Module):
    """
    """

    def __init__(self, learner_configuration, backbone_configuration):
        """

        :param learner_configuration: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = learner_configuration
        self.backbone_config = backbone_configuration

        self.vars = nn.ParameterList()
        self.static_plasticity = nn.ParameterList()
        self.context_plasticity_models = nn.ParameterList()
        self.attention = nn.ParameterList()
        self.context_plasticity_status = False
        self.static_plasticity_status = False
        self.meta_strength_status = False
        self.neuromodulation_status = False

        self.vars = self.parse_config(self.config, nn.ParameterList())
        self.context_backbone = self.parse_config(backbone_configuration, nn.ParameterList())

    def parse_config(self, config, vars_list):

        for i, info_dict in enumerate(config):
            if info_dict["name"] == 'conv2d':
                w, b = oml.nn.conv2d(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = oml.nn.linear(param_config["out"], param_config["in"], info_dict["adaptation"],
                                     info_dict["meta"])

                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == "linear-sparse":
                param_config = info_dict["config"]
                w, b = oml.nn.linear_sparse(param_config["out"], param_config["in"], param_config["spread"],
                                            info_dict["adaptation"],
                                            info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)
            #
            elif info_dict["name"] in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    #
    def add_static_plasticity(self):
        self.static_plasticity_status = True
        self.static_plasticity = nn.ParameterList()
        for weight in self.vars:
            if weight.adaptation:
                self.static_plasticity.append(oml.nn.static_plasticity(np.prod(weight.shape)))

    def add_meta_strength(self):
        #
        self.meta_strength_status = True
        self.meta_strength = nn.ParameterList()
        for weight in self.vars:
            if weight.adaptation:
                self.meta_strength.append(oml.nn.local_connectivity(weight))

    def add_neuromodulation(self, context_dimension):
        self.neuromodulation_status = True
        self.neuromodulation_parameters = nn.ParameterList()
        for weight in self.vars:
            w, b = oml.nn.linear(np.prod(weight.shape), context_dimension, adaptation=False, meta=True)
            self.neuromodulation_parameters.append(w)
            self.neuromodulation_parameters.append(b)

    def add_context_plasticity(self, context_dimension):
        self.context_plasticity_status = True
        self.context_plasticity_models = nn.ParameterList()
        for weight in self.vars:
            if weight.adaptation:
                w, b = oml.nn.linear(np.prod(weight.shape), context_dimension, adaptation=False, meta=True)
                self.context_plasticity_models.append(w)
                self.context_plasticity_models.append(b)

    def reset_vars(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
        :return:
        """
        for var in self.vars:
            if var.adaptation is True:
                if len(var.shape) > 1:
                    logger.info("Resetting weight")
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)

    def forward_plasticity(self, x):

        vars = self.context_plasticity_models

        x = self.forward(x, self.context_backbone, self.backbone_config)
        x = torch.mean(x, 0, keepdim=True)

        list_of_activations = []

        for i in range(0, len(vars), 2):
            w, b = vars[i], vars[i + 1]
            # print("Weight shape", self.vars[int(i/2)].shape)
            gating = F.relu(F.linear(x, w, b)).view(self.vars[int(i / 2)].shape).unsqueeze(0).unsqueeze(0)
            if len(gating.shape) == 4:
                gating = F.avg_pool2d(gating, kernel_size=(5, 5), stride=1, padding=2, count_include_pad=False)
            elif len(gating.shape) == 3:
                gating = F.avg_pool1d(gating, kernel_size=5, stride=1, padding=2, count_include_pad=False)

            list_of_activations.append(torch.mean(gating, 0))
        #
        return list_of_activations

    def forward(self, x, vars=None, config=None):
        """
        """

        neuromodulate = self.neuromodulation_status and config is None
        if neuromodulate:
            x_embedding = self.forward(x, vars=self.context_backbone, config=self.backbone_config)
        x = x.float()
        if vars is None:
            vars = self.vars

        flag = False
        if config is None:
            flag = True
            config = self.config

        idx = 0
        neuro_idx = 0
        #
        if self.meta_strength_status and flag:
            new_fast_weights = []
            for counter in range(len(vars)):
                temp = vars[counter] * self.meta_strength[counter].view(vars[counter].shape)
                new_fast_weights.append(temp)

            vars = new_fast_weights

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]

            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=info_dict['stride'], padding=info_dict['padding'])
                idx += 2

            elif name == 'linear-sparse':
                w, b = vars[idx], vars[idx + 1]
                w = w.T.unsqueeze(0)
                x = F.pad(x, (int(info_dict["config"]["spread"] / 2), int(info_dict["config"]["spread"] / 2) - 1))
                x = x.unfold(1, int(info_dict["config"]["spread"]), step=1)
                print("Explanded w shape", w.expand(x.shape[0], -1, -1).shape, x.shape)

                ## Einsum version uses noticeably more GPU memory  ¯\_(ツ)_/¯
                # x = torch.einsum("aib, aib -> ai", x, w.expand(x.shape[0], -1, -1)) + b.unsqueeze(0)
                x = (x * w).sum(2) + b.unsqueeze(0)

                idx += 2
                if neuromodulate:
                    w, b = self.neuromodulation_parameters[neuro_idx], self.neuromodulation_parameters[neuro_idx + 1]
                    x_gate = torch.sigmoid(F.linear(x_embedding, w, b)).unsqueeze(1)
                    assert (len(x_gate.shape) == 3)
                    # x_gate = F.avg_pool1d(x_gate, kernel_size=3, stride=1, padding=1, count_include_pad=False)
                    x_gate = x_gate.view(x.shape)
                    x = x * x_gate
                    neuro_idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]

                if neuromodulate:
                    w = w
                    w_weight, b_weight = self.neuromodulation_parameters[neuro_idx], self.neuromodulation_parameters[
                        neuro_idx + 1]
                    w_bias, b_bias = self.neuromodulation_parameters[neuro_idx + 2], self.neuromodulation_parameters[
                        neuro_idx + 3]
                    # print(x_embedding.shape, w_weight.shape, b_bias.shape)
                    #
                    weight_mask = torch.sigmoid(
                        F.linear(x_embedding, w_weight, b_weight).view(-1, w.shape[0], w.shape[1]))
                    bias_mask = torch.sigmoid(F.linear(x_embedding, w_bias, b_bias).view(-1, b.shape[0]))
                    w, b = w.unsqueeze(0), b.unsqueeze(0)
                    w = w * weight_mask
                    b = b * bias_mask
                    x = x.unsqueeze(2)
                    x = torch.bmm(w, x).squeeze(2) + b
                    # x_gate = torch.sigmoid(F.linear(x_embedding, w, b)).unsqueeze(1)
                    # assert (len(x_gate.shape) == 3)
                    # # x_gate = F.avg_pool1d(x_gate, kernel_size=3, stride=1, padding=1, count_include_pad=False)
                    # x_gate = x_gate.view(x.shape)
                    # x = x * x_gate
                    neuro_idx += 4
                else:
                    x = F.linear(x, w, b)
                idx += 2


            elif name == 'flatten':

                x = x.view(x.size(0), -1)

            elif name == 'reshape':
                continue

            elif name == 'relu':
                x = F.relu(x)

            elif name == 'tanh':
                x = F.tanh(x)

            elif name == 'sigmoid':
                x = torch.sigmoid(x)

            else:
                raise NotImplementedError
        assert idx == len(vars)
        return x

    def update_weights(self, vars):

        for old, new in zip(self.vars, vars):
            old.data = new.data

    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))

    def get_forward_meta_parameters(self):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        return list(filter(lambda x: x.meta, list(self.vars)))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars
