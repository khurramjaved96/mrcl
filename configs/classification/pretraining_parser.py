import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()

        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--path', help='Path of the dataset', default="../")
        self.add('--epoch', type=int, nargs='+', help='epoch number', default=[45])
        self.add('--dataset', help='Name of experiment', default="omniglot")
        self.add('--lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.0001])
        self.add('--name', help='Name of experiment', default="baseline")

