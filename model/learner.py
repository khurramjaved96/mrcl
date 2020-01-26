import logging

logger = logging.getLogger("experiment")

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("experiment")


#

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

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        self.meta_plasticity = nn.ParameterList()
        self.context_models = nn.ParameterList()
        self.neuromodulation = nn.ParameterList()
        # self.meta_vars = nn.ParameterList()
        self.modulate = False
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        # self.meta_vars_bn = nn.ParameterList()
        store_meta = False

        for i, (name, adaptation, meta_param, param) in enumerate(self.config):
            # print("Name = ", name)
            if i == 0:
                input_param = param
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                w.learn = adaptation
                self.vars.append(w)
                # [ch_out]
                b = nn.Parameter(torch.zeros(param[0]))
                b.learn = adaptation
                self.vars.append(b)

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                w.learn = adaptation
                # gain=1 according to cbfin's implementation

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':

                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                w.learn = adaptation
                w.meta = meta_param
                torch.nn.init.kaiming_normal_(w)
                b = nn.Parameter(torch.zeros(param[0]))
                b.learn = adaptation
                b.meta = meta_param

                self.vars.append(w)
                self.vars.append(b)

                w_mod = nn.Parameter(torch.ones(param[0], input_param[1]))
                w_mod.learn = False
                torch.nn.init.kaiming_normal_(w_mod)
                b_mod = nn.Parameter(torch.zeros(param[0]))
                b_mod.learn = False
                w_mod.meta = True
                b_mod.meta = True

                self.neuromodulation.append(w_mod)
                self.neuromodulation.append(b_mod)

                w_context_plasticity = nn.Parameter(torch.ones(param[0], input_param[1]))
                w_context_plasticity.learn = False
                torch.nn.init.kaiming_normal_(w_context_plasticity)
                b_context_plasticity = nn.Parameter(torch.zeros(param[0]))
                b_context_plasticity.learn = False
                w_context_plasticity.meta = True
                b_context_plasticity.meta = True

                self.context_models.append(w_context_plasticity)
                self.context_models.append(b_context_plasticity)

                if adaptation:
                    w = nn.Parameter(torch.zeros(*param))
                    # torch.nn.init.zeros_(w)
                    # torch.nn.init.normal_(w)
                    # w = w*100
                    self.meta_plasticity.append(w)
                    b = nn.Parameter(torch.zeros(param[0]))
                    # torch.nn.init.normal_(b)
                    # b = b*100
                    self.meta_plasticity.append(b)
                    w.learn = False
                    b.learn = False
                    w.meta = True
                    b.meta = True

            #

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
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                w.learn = adaptation
                self.vars.append(w)
                # [ch_out]
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
                raise NotImplementedError


    def decay_plasticity(self, decay_rate):
        for param in self.meta_plasticity:
            param = param * decay_rate

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


    def forward_plasticity(self, x):

        x = x.float()

        vars = self.context_models

        idx = 0
        bn_idx = 0
        list_of_activations = []
        for name, meta, meta_param, param in self.config:

            if name == 'conv2d':
                assert(False)
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])


            elif name == 'convt2d':
                assert (False)
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2

            elif name == 'linear':

                w, b = vars[idx], vars[idx + 1]
                gating = F.linear(x, w, b)
                list_of_activations.append(torch.mean(F.relu(gating),0))
                idx += 2
            #
            elif name == 'rep':
                pass


            elif name == 'flatten':

                pass


            elif name == 'reshape':

                pass


            elif name == 'relu':
                pass

            elif name == "modulate":
                pass


            elif name == 'leakyrelu':
                pass

            elif name == 'tanh':
                pass
            elif name == 'sigmoid':
                pass
            elif name == 'upsample':
                pass

            elif name == 'max_pool2d':
                pass

            elif name == 'avg_pool2d':
                pass

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                # print("BN weifght = ", w)
                # print("BN bias = ", b)
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=False)
                idx += 2
                bn_idx += 2

            else:
                raise NotImplementedError

        # make sure variable is used properly
        # list_of_activations.append(torch.mean(x, dim=0))
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return list_of_activations



    def forward(self, x, vars=None, bn_training=True, feature=False):
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

        x_input = x
        idx = 0
        bn_idx = 0
        #

        for name, meta, meta_param, param in self.config:

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
                x = F.linear(x, w, b)


                idx += 2
            #
            elif name == 'rep':
                # quit()
                # print(x.shape)
                if feature:
                    return x


            elif name == 'flatten':

                # print(x.shape)

                x = x.view(x.size(0), -1)


            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)


            elif name == 'relu':
                x = F.relu(x, inplace=param[0])


            elif name == "modulate":
                w, b = neuro_mod[idx-2], neuro_mod[idx - 1]
                x_mod = F.relu(F.linear(x_input, w, b))
                x = x * x_mod


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

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
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
