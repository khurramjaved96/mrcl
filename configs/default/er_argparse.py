import configargparse


class ERParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/default/empty.ini",
                 help='config file path')

        self.add('--epoch', type=int, help='epoch number', default=1)
        self.add('--seed', nargs='+', help='Seed', default=[60, 70], type=int)
        self.add('--tasks', type=int, help='meta batch size, namely task num', default=1)
        self.add('--lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.01, 0.001, 0.0003])
        self.add('--name', help='Name of experiment', default="oml_classification")
        self.add('--dataset', help='Name of experiment', default="omniglot")
        self.add("--commit", action="store_true")
        self.add("--mask", action="store_true", default=True)
        self.add('--dataset-path', help='Name of experiment', default=None)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--memory', type=int, help='meta batch size, namely task num', default=400)



        