import logging

logger = logging.getLogger("experiment")

import logging

import torch
from torch import nn
from torch.nn import functional as F
import model.initialize_layers as layer_init

logger = logging.getLogger("experiment")


class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.full_config = config

        self.config = config

        self.plasticity_config = []
        self.neuromodulation_config = []

        self.vars = nn.ParameterList()
        self.meta_plasticity = nn.ParameterList()
        self.context_models = nn.ParameterList()
        self.neuromodulation = nn.ParameterList()

        self.modulate = False
        self.vars_bn = nn.ParameterList()
        self.plasticity_backbone = nn.ParameterList()
        self.neuromodulation_backbone = nn.ParameterList()
        self.context_backbone_width = 51
        self.neuromodulation_backbone_width = 51
        self.attention = nn.ParameterList()
        self.attention_span = 16

        for i, (name, adaptation, meta_param, param) in enumerate(self.config):

            if "plasticity-bone" in name:
                #     Do plasticity stuff
                if name.split("-")[0] in "linear":
                    w = nn.Parameter(torch.ones(*param))
                    w.learn = adaptation
                    w.meta = meta_param
                    torch.nn.init.kaiming_normal_(w)
                    b = nn.Parameter(torch.zeros(param[0]))
                    b.learn = adaptation
                    b.meta = meta_param
                    self.plasticity_backbone.append(w)
                    self.plasticity_backbone.append(b)
                    self.context_backbone_width = param[0]

            elif "neuromodulation-bone" in name:
                if name.split("-")[0] in "linear":
                    w = nn.Parameter(torch.ones(*param))
                    w.learn = adaptation
                    w.meta = meta_param
                    torch.nn.init.kaiming_normal_(w)
                    b = nn.Parameter(torch.zeros(param[0]))
                    b.learn = adaptation
                    b.meta = meta_param
                    self.neuromodulation_backbone.append(w)
                    self.neuromodulation_backbone.append(b)
                    self.neuromodulation_backbone_width = param[0]

            elif name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                w.learn = adaptation
                self.vars.append(w)
                b = nn.Parameter(torch.zeros(param[0]))
                b.learn = adaptation
                self.vars.append(b)

            elif name is 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                w.learn = adaptation

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'attention':

                w_values, w_values_bias, w_keys, w_keys_bias = layer_init.initialize_attention(param[0], param[1],
                                                                                               self.neuromodulation_backbone_width,
                                                                                               self.attention_span,
                                                                                               adaptation, meta_param)

                self.vars.append(w_values)
                self.vars.append(w_values_bias)

                self.attention.append(w_keys)
                self.attention.append(w_keys_bias)

                if adaptation:
                    w_context_plasticity, b_context_plasticity = layer_init.initialize_context_plasticity(
                        param[0] * param[1] * self.attention_span, self.context_backbone_width)

                    self.context_models.append(w_context_plasticity)
                    self.context_models.append(b_context_plasticity)

                    w_context_plasticity_bias, b_context_plasticity_bias = layer_init.initialize_context_plasticity(
                        param[0] * self.attention_span, self.context_backbone_width)

                    self.context_models.append(w_context_plasticity_bias)
                    self.context_models.append(b_context_plasticity_bias)

                    w, b = layer_init.initialize_plasticity(param[0] * self.attention_span, param[1])

                    self.meta_plasticity.append(w)
                    self.meta_plasticity.append(b)

            elif name == "positional-attention":
                w_values, w_values_bias, w_query, w_query_bias = layer_init.initialize_positional_attention(param[0],
                                                                                                            param[1],
                                                                                                            self.neuromodulation_backbone_width,
                                                                                                            self.attention_span,
                                                                                                            adaptation,
                                                                                                            meta_param)
                #
                self.vars.append(w_values)
                self.vars.append(w_values_bias)

                self.attention.append(w_query)
                self.attention.append(w_query_bias)
                #
                if adaptation:
                    w_context_plasticity, b_context_plasticity = layer_init.initialize_context_plasticity(
                        param[0] * param[1] * self.attention_span, self.context_backbone_width)

                    self.context_models.append(w_context_plasticity)
                    self.context_models.append(b_context_plasticity)

                    w_context_plasticity_bias, b_context_plasticity_bias = layer_init.initialize_context_plasticity(
                        param[0] * self.attention_span, self.context_backbone_width)

                    self.context_models.append(w_context_plasticity_bias)
                    self.context_models.append(b_context_plasticity_bias)

                    w, b = layer_init.initialize_plasticity(param[0] * self.attention_span, param[1])

                    self.meta_plasticity.append(w)
                    self.meta_plasticity.append(b)

            elif name is 'linear':

                w, b = layer_init.initialize_linear(param[0], param[1], adaptation, meta_param)

                self.vars.append(w)
                self.vars.append(b)

                if adaptation:
                    w_context_plasticity, b_context_plasticity = layer_init.initialize_context_plasticity(
                        param[0] * param[1],
                        self.context_backbone_width)

                    self.context_models.append(w_context_plasticity)
                    self.context_models.append(b_context_plasticity)

                    w_context_plasticity_bias, b_context_plasticity_bias = layer_init.initialize_context_plasticity(
                        param[0],
                        self.context_backbone_width)

                    self.context_models.append(w_context_plasticity_bias)
                    self.context_models.append(b_context_plasticity_bias)

                    w, b = layer_init.initialize_plasticity(param[0], param[1])
                    self.meta_plasticity.append(w)
                    self.meta_plasticity.append(b)


            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                store_meta = True
                pass
            elif name is "gate":
                pass
            elif name is "modulate":
                self.modulate = True
            elif name is 'bn':

                w = nn.Parameter(torch.ones(param[0]))
                w.learn = adaptation
                self.vars.append(w)
                b = nn.Parameter(torch.zeros(param[0]))
                b.learn = adaptation
                self.vars.append(b)

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                print(name)
                raise NotImplementedError

        temp_config = []
        for i, (name, adaptation, meta_param, param) in enumerate(self.config):
            if "plasticity-bone" in name:
                self.plasticity_config.append(self.config[i])
            elif "neuromodulation-bone" in name:
                self.neuromodulation_config.append(self.config[i])
            else:
                temp_config.append(self.config[i])

        self.config = temp_config

        # self.log_model()

    def reset_vars(self):
        for a in self.vars:
            if len(a.shape) > 1:
                logger.info("Resetting weight")
                torch.nn.init.kaiming_normal_(a)

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"

            elif name is 'gate':
                tmp = 'gate'
                info += tmp + "\n"

            elif name is "modulate":
                tmp = 'modulate'
                info += tmp + "\n"
            elif name is "attention":
                tmp = "attention"
                info += tmp + "\n"

            elif name is "positional-attention":
                tmp = "positional-attention"
                info += tmp + "\n"

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward_plasticity(self, x, log=False):

        x = x.float()

        x = torch.mean(x, 0, keepdim=True)

        # Do plasticity forward pass

        idx_plasticity = 0

        if log:
            logger.debug("Plasticity embedding = x")
            logger.debug("X shape = %s", str(x.shape))
        plasticity_embedding = x
        self.plasticity_config = []
        for name, meta, meta_param, param in self.plasticity_config:
            name = name.split("-")[0]
            if name == 'linear':

                w, b = self.plasticity_backbone[idx_plasticity], self.plasticity_backbone[idx_plasticity + 1]
                plasticity_embedding = F.linear(plasticity_embedding, w, b)
                idx_plasticity += 2
                if log:
                    logger.debug("Backbone Linear transformation of plasticity embedding using param %s %s",
                                 str(w.shape), str(b.shape))

            elif name == "relu":
                plasticity_embedding = F.relu(plasticity_embedding)
                if log:
                    logger.debug("Relu plastic embedding")
            else:
                assert (False)
        if log:
            logger.debug("Plasticity embedding computing\n")
        vars = self.context_models

        idx = 0
        bn_idx = 0
        list_of_activations = []
        for name, meta, meta_param, param in self.config:

            if name == "attention":
                w, b = vars[idx], vars[idx + 1]
                gating = F.linear(plasticity_embedding, w, b)
                list_of_activations.append(torch.mean(F.relu(gating), 0))

                w_bias, b_bias = vars[idx + 2], vars[idx + 3]
                gating_bias = F.linear(plasticity_embedding, w_bias, b_bias)
                list_of_activations.append(torch.mean(F.relu(gating_bias), 0))
                #
                if log:
                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-2].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w.shape), str(b.shape))

                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-1].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w_bias.shape), str(b_bias.shape))

                idx += 4

            if name == "positional-attention":
                w, b = vars[idx], vars[idx + 1]
                gating = F.linear(plasticity_embedding, w, b)
                list_of_activations.append(torch.mean(F.relu(gating), 0))

                w_bias, b_bias = vars[idx + 2], vars[idx + 3]
                gating_bias = F.linear(plasticity_embedding, w_bias, b_bias)
                list_of_activations.append(torch.mean(F.relu(gating_bias), 0))
                #
                if log:
                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-2].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w.shape), str(b.shape))

                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-1].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w_bias.shape), str(b_bias.shape))

                idx += 4

            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                gating = F.linear(plasticity_embedding, w, b)
                list_of_activations.append(torch.mean(F.relu(gating), 0))

                w_bias, b_bias = vars[idx + 2], vars[idx + 3]
                gating_bias = F.linear(plasticity_embedding, w_bias, b_bias)
                list_of_activations.append(torch.mean(F.relu(gating_bias), 0))
                #
                if log:
                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-2].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w.shape), str(b.shape))

                    logger.debug("Linear transformation of embedding to get shape %s",
                                 str(list_of_activations[-1].shape))
                    logger.debug("Linear transformation weights shape %s %s", str(w_bias.shape), str(b_bias.shape))
                #
                idx += 4
        if log:
            logger.debug("\n")
        # make sure variable is used properly
        # list_of_activations.append(torch.mean(x, dim=0))
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return list_of_activations

    def forward(self, x, vars=None, bn_training=True, feature=False, log=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        x = x.float()
        if vars is None:
            vars = self.vars

        neuro_mod = self.neuromodulation

        x_attention = x
        # x_input = torch.mean(x, dim=0, keepdim=True)
        idx_neuro = 0
        # if log:
        #     logger.debug("\nNeuromodulation embedding = x with shape %s", str(x_input.shape))

        self.neuromodulation_config = []
        for name, meta, meta_param, param in self.neuromodulation_config:
            name = name.split("-")[0]
            if name == 'linear':

                w, b = self.neuromodulation_backbone[idx_neuro], self.plasticity_backbone[idx_neuro + 1]
                x_input = F.linear(x_input, w, b)
                if log:
                    logger.debug("Backbone Linear transformation to get shape %s", str(x_input.shape))
                idx_neuro += 2

            elif name == "relu":
                if log:
                    logger.debug("Backbone Relu")
                x_input = F.relu(x_input)

            else:
                assert (False)

        # Do a forward pass of modulation network here

        idx = 0
        mod_id = 0
        bn_idx = 0
        key_id = 0
        #
        x_key = x
        for layer_counter, (name, meta, meta_param, param) in enumerate(self.config):

            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2

                # print(name, param, '\tout:', x.shape)

            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2

            elif name == 'linear':

                w, b = vars[idx], vars[idx + 1]

                # If equal, then last layer
                if layer_counter != (len(self.config) - 1):
                    if self.config[layer_counter + 1][0] is "modulate":
                        w_w, b_w = self.neuromodulation[idx_neuro], self.neuromodulation[idx_neuro + 1]
                        w_b, b_b = self.neuromodulation[idx_neuro + 2], self.neuromodulation[idx_neuro + 3]
                        weight_mask = F.relu(F.linear(x_input, w_w, b_w))
                        bias_mask = F.relu(F.linear(x_input, w_b, b_b))

                        weight_mask = weight_mask.view(w.shape)
                        bias_mask = bias_mask.view(b.shape)
                        if log:
                            logger.debug("Weight mask = %s", str(weight_mask.shape))
                            logger.debug("Bias mask = %s", str(bias_mask.shape))
                            logger.debug("Applying weight and base mask for modulation")
                            logger.debug("Weight shape = %s", str(w.shape))
                        w = w * weight_mask
                        b = b * bias_mask
                        idx_neuro += 4

                x = F.linear(x, w, b)
                if log:
                    logger.debug("Applying linear transformation to get shape %s", str(x.shape))
                    logger.debug("Weights used %s %s", str(w.shape), str(b.shape))
                idx += 2

            elif name == "attention":
                #
                w, b = vars[idx], vars[idx + 1]
                # print(x.shape, w.shape, b.shape)
                values = F.linear(x, w, b).view(x.shape[0], -1, self.attention_span)

                query = x_attention.view(x.shape[0], 11, 1).expand(-1, -1, 10)

                w_key, b_key = self.attention[key_id], self.attention[key_id + 1]
                #
                # print(b_key)
                keys = F.linear(x, w_key, b_key).view(x.shape[0], 11, self.attention_span)
                #
                # print(query.shape, keys.shape)
                query_key = F.cosine_similarity(query, keys, dim=1).unsqueeze(1)
                # print(query_key.shape)
                # print(query_key[0, :])
                # query_key = torch.div(torch.matmul(query, keys), 3.31)
                # query_key = torch.matmul(query, keys)
                #
                # print(query_key)
                softmax_key = torch.softmax(query_key, dim=2)
                # print(softmax_key[0, :])
                # if key_id == 0:
                #     print(name)
                #     for a in softmax_key:
                #         print(a)
                #         break
                #     print(softmax_key.shape)
                # print(softmax_key)

                #
                #
                # x_attention_temp = x_attention[:, 0:10].unsqueeze(1)
                # print(x_attention_temp.shape, values.shape)
                values_softened = values * softmax_key
                keys = keys * softmax_key

                x_key = torch.sum(keys, 2)

                # values_softened = values * x_attention_temp
                # #
                values_softened = torch.sum(values_softened, 2)
                # values_softened = torch.mean(values, 2)
                #
                #
                if log:
                    logger.debug("Keys shape = %s", str(keys.shape))
                    logger.debug("Query shape = %s", str(query.shape))
                    logger.debug("Query key shape = %s", str(query_key.shape))
                    logger.debug("Values shape = %s", str(values.shape))
                    logger.debug("Softmax shape = %s", str(softmax_key.shape))
                    logger.debug("Final shape = %s", str(values_softened.shape))

                # values_softened = values[:, :, 0].squeeze()

                x = values_softened
                key_id += 2
                idx += 2

            elif name == "positional-attention":
                #
                w, b = vars[idx], vars[idx + 1]
                values = F.linear(x, w, b).view(x.shape[0], -1, self.attention_span)

                query = x_attention
                w_key, b_key = self.attention[key_id], self.attention[key_id + 1]

                keys = F.linear(query, w_key, b_key).unsqueeze(1)

                softmax_key = torch.softmax(keys, dim=2)
                # print(softmax_key.shape, values.shape)
                values_softened = values * softmax_key

                values_softened = torch.sum(values_softened, 2)

                if log:
                    logger.debug("Keys shape = %s", str(keys.shape))
                    logger.debug("Query shape = %s", str(query.shape))
                    logger.debug("Values shape = %s", str(values.shape))
                    logger.debug("Softmax shape = %s", str(softmax_key.shape))
                    logger.debug("Final shape = %s", str(values_softened.shape))

                x = values_softened
                key_id += 2
                idx += 2



            elif name == 'rep':
                # quit()
                # print(x.shape)
                if feature:
                    return x
            #

            elif name == 'flatten':

                # print(x.shape)

                x = x.view(x.size(0), -1)


            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)


            elif name == 'relu':
                if log:
                    logger.debug("Relu x")
                x = F.relu(x)


            elif name == "modulate":
                pass
                # w, b = neuro_mod[idx - 2], neuro_mod[idx - 1]
                # x_mod = F.relu(F.linear(x_input, w, b))
                # x = x * x_mod
                # if log:
                #     logger.debug("Modulating x. Modulation shape %s, x Shape %s", str(x_mod.shape), str(x.shape))
                # # assert(False)


            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])


            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                # print("BN weifght = ", w)
                # print("BN bias = ", b)
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x

    def log_model(self):
        for i, (name, adaptation, meta_param, param) in enumerate(self.config):
            logger.info("Model Params = %s", name)
        for i, (name, adaptation, meta_param, param) in enumerate(self.neuromodulation_config):
            logger.info("Neuromodulation Backbone = %s", name)
        for i, (name, adaptation, meta_param, param) in enumerate(self.plasticity_config):
            logger.info("Context backbone = %s", name)

    #
    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        assert (False)
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """

        return self.vars

# class Learner(nn.Module):
#     """
#
#     """
#
#     def __init__(self, config, meta=False):
#         """
#
#         :param config: network config file, type:list of (string, list)
#         :param imgc: 1 or 3
#         :param imgsz:  28 or 84
#         """
#         super(Learner, self).__init__()
#
#         self.config = config
#
#         # this dict contains all tensors needed to be optimized
#         self.vars = nn.ParameterList()
#         self.meta_vars = nn.ParameterList()
#         # self.meta_vars = nn.ParameterList()
#         # running_mean and running_var
#         self.vars_bn = nn.ParameterList()
#         # self.meta_vars_bn = nn.ParameterList()
#         store_meta = False
#         for i, (name, param) in enumerate(self.config):
#             if name is 'conv2d':
#                 # [ch_out, ch_in, kernelsz, kernelsz]
#                 w = nn.Parameter(torch.ones(*param[:4]))
#                 # gain=1 according to cbfin's implementation
#                 if meta:
#                     torch.nn.init.zeros_(w)
#                 else:
#                     torch.nn.init.kaiming_normal_(w)
#                 self.vars.append(w)
#                 # [ch_out]
#                 self.vars.append(nn.Parameter(torch.zeros(param[0])))
#
#             elif name is 'convt2d':
#                 # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
#                 w = nn.Parameter(torch.ones(*param[:4]))
#                 # gain=1 according to cbfin's implementation
#                 if meta:
#                     torch.nn.init.zeros_(w)
#                 else:
#                     torch.nn.init.kaiming_normal_(w)
#                 self.vars.append(w)
#                 # [ch_in, ch_out]
#                 self.vars.append(nn.Parameter(torch.zeros(param[1])))
#
#             elif name is 'linear':
#
#                 # [ch_out, ch_in]
#                 w = nn.Parameter(torch.ones(*param))
#                 # gain=1 according to cbfinn's implementation
#
#                 torch.nn.init.kaiming_normal_(w)
#                 self.vars.append(w)
#                 # [ch_out]
#                 self.vars.append(nn.Parameter(torch.zeros(param[0])))
#                 if store_meta:
#                     w = nn.Parameter(torch.ones(*param))
#                     # torch.nn.init.zeros_(w)
#                     torch.nn.init.kaiming_normal_(w)
#                     self.meta_vars.append(w)
#                     self.meta_vars.append(nn.Parameter(torch.zeros(param[0])))
#
#
#
#             elif name is 'cat':
#                 pass
#             elif name is 'cat_start':
#                 pass
#             elif name is "rep":
#                 store_meta = True
#                 pass
#             elif name is "gate":
#                 pass
#             elif name is 'bn':
#                 # [ch_out]
#                 w = nn.Parameter(torch.ones(param[0]))
#                 self.vars.append(w)
#                 # [ch_out]
#                 self.vars.append(nn.Parameter(torch.zeros(param[0])))
#
#                 # must set requires_grad=False
#                 running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
#                 running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
#                 self.vars_bn.extend([running_mean, running_var])
#
#
#             elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
#                           'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
#                 continue
#             else:
#                 raise NotImplementedError
#
#
#         # for i, (name, param) in enumerate(self.config):
#         #     if name is 'conv2d':
#         #         # [ch_out, ch_in, kernelsz, kernelsz]
#         #         w = nn.Parameter(torch.ones(*param[:4]))
#         #         # gain=1 according to cbfin's implementation
#         #         torch.nn.init.kaiming_normal_(w)
#         #         self.meta_vars.append(w)
#         #         # [ch_out]
#         #         self.meta_vars.append(nn.Parameter(torch.zeros(param[0])))
#         #
#         #     elif name is 'convt2d':
#         #         # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
#         #         w = nn.Parameter(torch.ones(*param[:4]))
#         #         # gain=1 according to cbfin's implementation
#         #         torch.nn.init.kaiming_normal_(w)
#         #         self.meta_vars.append(w)
#         #         # [ch_in, ch_out]
#         #         self.meta_vars.append(nn.Parameter(torch.zeros(param[1])))
#         #
#         #     elif name is 'linear':
#         #
#         #         # [ch_out, ch_in]
#         #         w = nn.Parameter(torch.ones(*param))
#         #         # gain=1 according to cbfinn's implementation
#         #         torch.nn.init.kaiming_normal_(w)
#         #         self.meta_vars.append(w)
#         #         # [ch_out]
#         #         self.meta_vars.append(nn.Parameter(torch.zeros(param[0])))
#         #
#         #     elif name is 'cat':
#         #         pass
#         #     elif name is 'gate':
#         #         pass
#         #     elif name is 'cat_start':
#         #         pass
#         #     elif name is "rep":
#         #         pass
#         #     elif name is 'bn':
#         #         # [ch_out]
#         #         w = nn.Parameter(torch.ones(param[0]))
#         #         self.meta_vars.append(w)
#         #         # [ch_out]
#         #         self.meta_vars.append(nn.Parameter(torch.zeros(param[0])))
#         #
#         #         # must set requires_grad=False
#         #         running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
#         #         running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
#         #         self.meta_vars_bn.extend([running_mean, running_var])
#         #
#         #
#         #     elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
#         #                   'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
#         #         continue
#         #     else:
#         #         raise NotImplementedError
#
#     def extra_repr(self):
#         info = ''
#
#         for name, param in self.config:
#             if name is 'conv2d':
#                 tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
#                       % (param[1], param[0], param[2], param[3], param[4], param[5],)
#                 info += tmp + '\n'
#
#             elif name is 'convt2d':
#                 tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
#                       % (param[0], param[1], param[2], param[3], param[4], param[5],)
#                 info += tmp + '\n'
#
#             elif name is 'linear':
#                 tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
#                 info += tmp + '\n'
#
#             elif name is 'leakyrelu':
#                 tmp = 'leakyrelu:(slope:%f)' % (param[0])
#                 info += tmp + '\n'
#
#             elif name is 'cat':
#                 tmp = 'cat'
#                 info += tmp + "\n"
#             elif name is 'cat_start':
#                 tmp = 'cat_start'
#                 info += tmp + "\n"
#
#             elif name is 'rep':
#                 tmp = 'rep'
#                 info += tmp + "\n"
#
#             elif name is 'gate':
#                 tmp = 'gate'
#                 info += tmp + "\n"
#
#
#             elif name is 'avg_pool2d':
#                 tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
#                 info += tmp + '\n'
#             elif name is 'max_pool2d':
#                 tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
#                 info += tmp + '\n'
#             elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
#                 tmp = name + ':' + str(tuple(param))
#                 info += tmp + '\n'
#             else:
#                 raise NotImplementedError
#
#         return info
#
#     def forward(self, x, vars=None, bn_training=True, feature=False):
#         """
#         This function can be called by finetunning, however, in finetunning, we dont wish to update
#         running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
#         Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
#         but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
#         :param x: [b, 1, 28, 28]
#         :param vars:
#         :param bn_training: set False to not update
#         :return: x, loss, likelihood, kld
#         """
#
#         if vars is None:
#             vars = self.vars
#
#         # meta_vars = self.meta_vars
#
#
#         x_meta = x
#         idx = 0
#         bn_idx = 0
#
#         for name, param in self.config:
#
#             if name == 'conv2d':
#                 w, b = vars[idx], vars[idx + 1]
#                 x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
#
#                 # w_meta, b_meta = meta_vars[idx], meta_vars[idx + 1]
#                 # x_meta = F.conv2d(x_meta, w_meta, b_meta, stride=param[4], padding=param[5])
#
#                 idx += 2
#
#
#                 # print(name, param, '\tout:', x.shape)
#
#             elif name == 'convt2d':
#                 w, b = vars[idx], vars[idx + 1]
#                 x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
#
#                 # w_meta, b_meta = meta_vars[idx], meta_vars[idx + 1]
#                 # x_meta = F.conv_transpose2d(x_meta, w_meta, b_meta, stride=param[4], padding=param[5])
#
#
#                 idx += 2
#
#             elif name == 'linear':
#
#                 w, b = vars[idx], vars[idx + 1]
#                 x = F.linear(x, w, b)
#
#                 # w_meta, b_meta = meta_vars[idx], meta_vars[idx + 1]
#                 # x_meta = F.linear(x_meta, w_meta, b_meta)
#
#
#
#                 idx += 2
#
#             elif name == 'rep':
#                 # quit()
#                 # print(x.shape)
#                 if feature:
#                     return x
#
#
#             elif name == 'flatten':
#
#                 # print(x.shape)
#
#                 x = x.view(x.size(0), -1)
#                 # x_meta = x_meta.view(x_meta.size(0), -1)
#
#             elif name == 'reshape':
#                 # [b, 8] => [b, 2, 2, 2]
#                 x = x.view(x.size(0), *param)
#                 # x_meta = x_meta.view(x_meta.size(0), *param)
#
#             elif name == 'relu':
#                 x = F.relu(x, inplace=param[0])
#                 # x_meta = F.relu(x_meta, inplace=param[0])
#
#             elif name == "gate":
#                 x = x*x_meta
#                 assert(False)
#
#             elif name == 'leakyrelu':
#                 x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
#
#             elif name == 'tanh':
#                 x = F.tanh(x)
#             elif name == 'sigmoid':
#                 x = torch.sigmoid(x)
#             elif name == 'upsample':
#                 x = F.upsample_nearest(x, scale_factor=param[0])
#
#             elif name == 'max_pool2d':
#                 x = F.max_pool2d(x, param[0], param[1], param[2])
#
#                 # x_meta = F.max_pool2d(x_meta, param[0], param[1], param[2])
#             elif name == 'avg_pool2d':
#                 x = F.avg_pool2d(x, param[0], param[1], param[2])
#
#                 # x_meta = F.avg_pool2d(x_meta, param[0], param[1], param[2])
#
#             elif name == 'bn':
#                 w, b = vars[idx], vars[idx + 1]
#                 # print("BN weifght = ", w)
#                 # print("BN bias = ", b)
#                 running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
#                 x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
#                 idx += 2
#                 bn_idx += 2
#
#             else:
#                 raise NotImplementedError
#
#         # make sure variable is used properly
#         assert idx == len(vars)
#         assert bn_idx == len(self.vars_bn)
#
#
#         return x
#
#     def zero_grad(self, vars=None):
#         """
#
#         :param vars:
#         :return:
#         """
#         with torch.no_grad():
#             if vars is None:
#                 for p in self.vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#             else:
#                 for p in vars:
#                     if p.grad is not None:
#                         p.grad.zero_()
#
#     def parameters(self):
#         """
#         override this function since initial parameters will return with a generator.
#         :return:
#         """
#         return self.vars
