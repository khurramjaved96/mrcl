from torchmeta.modules import (MetaModule, MetaLinear)
from torchmeta.modules.batchnorm import MetaBatchNorm1d
from torchmeta.modules.utils import get_subdict


class PLN(MetaModule):
    def __init__(self, in_features, out_features, hidden_size=300):
        super(PLN, self).__init__()

        self.fc1 = MetaLinear(in_features, hidden_size)
        # self.bn1 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc2 = MetaLinear(hidden_size, hidden_size)
        # self.bn2 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc3 = MetaLinear(hidden_size, hidden_size)
        # self.bn3 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc4 = MetaLinear(hidden_size, hidden_size)
        self.fc5 = MetaLinear(hidden_size, hidden_size)

        self.fc6 = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None, bn_training=False):
        # if params is not None:
        #     for key in params:
        #         print(key)
        x = self.fc1(inputs, params=get_subdict(params, 'fc1'))
        # x = self.bn1(x, bn_training=bn_training)

        x = self.fc2(x, params=get_subdict(params, 'fc2'))
        # x = self.bn2(x,  bn_training=bn_training)

        x = self.fc3(x, params=get_subdict(params, 'fc3'))
        # x = self.bn3(x, bn_training=bn_training)
        x = self.fc4(x, params=get_subdict(params, 'fc4'))
        x = self.fc5(x, params=get_subdict(params, 'fc5'))
        x = self.fc6(x, params=get_subdict(params, 'fc6'))

        return x


class Plasticity(MetaModule):
    def __init__(self, in_features, out_features, hidden_size=300):
        super(Plasticity, self).__init__()

        self.fc1 = MetaLinear(in_features, hidden_size)
        # self.bn1 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc2 = MetaLinear(hidden_size, hidden_size)
        # self.bn2 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc3 = MetaLinear(hidden_size, hidden_size)
        # self.bn3 = MetaBatchNorm1d(hidden_size, momentum=1.)
        self.fc4 = MetaLinear(hidden_size, hidden_size)
        self.fc5 = MetaLinear(hidden_size, hidden_size)

        self.fc6 = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        assert (False)
        x = self.fc1(inputs, params=get_subdict(params, 'fc1'))
        x = self.fc2(x, params=get_subdict(params, 'fc2'))
        x = self.fc3(x, params=get_subdict(params, 'fc3'))
        x = self.fc4(x, params=get_subdict(params, 'fc4'))
        return x
