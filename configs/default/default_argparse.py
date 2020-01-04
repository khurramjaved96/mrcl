import configargparse


class DefaultParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/default/default.ini",
                 help='config file path')

        self.add('--steps', type=int, help='epoch number', default=40000)
        self.add('--seed', nargs='+', help='Seed', default=[60, 70], type=int)
        self.add('--tasks', type=int, help='meta batch size, namely task num', default=1)
        self.add('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate', default=[0.003, 0.001, 0.0003])
        self.add('--update_step', type=int, help='task-level inner update steps', default=5)
        self.add('--name', help='Name of experiment', default="oml_classification")
        self.add('--dataset', help='Name of experiment', default="omniglot")
        self.add("--commit", action="store_true")
        self.add('--dataset-path', help='Name of experiment', default=None)
        self.add("--no-reset", action="store_true")
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)



        