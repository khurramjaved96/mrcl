import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')

        self.add('--epoch', type=int, help='epoch number', default=200000)
        self.add('--tasks', type=int, help='meta batch size, namely task num', default=10)
        self.add('--capacity', type=int, help='meta batch size, namely task num', default=10)
        self.add('--meta_lr', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-4])
        self.add('--plasticity_lr', nargs='+', type=float, help='meta-level outer learning rate', default=[0.3, 0.1, 0.03, 0.01, 0.003])
        self.add('--modulation_lr', nargs='+', type=float, help='meta-level outer learning rate',
                 default=[1e-4])
        self.add('--update_lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.003, 0.001])
        self.add('--update_step', type=int, help='task-level inner update steps', default=10)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--model', help='Name of model', default="maml")
        self.add('--model-path', help='Path to model', default=None)
        self.add("--second-order", action="store_true")
        self.add("--no-plasticity", action="store_true")
        self.add("--no-sigmoid", action="store_true")
        self.add("--no-neuro", action="store_true")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", type=int, default=400)
        self.add("--base", type=str)
#
