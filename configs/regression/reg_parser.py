import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/regression/empty.ini",
                 help='config file path')

        self.add('--epoch', type=int, help='epoch number', default=300000)
        self.add('--tasks', type=int, help='meta batch size, namely task num', default=10)
        self.add('--capacity', type=int, help='meta batch size, namely task num', default=50)
        self.add('--meta-lr', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-4])
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--plasticity-lr', nargs='+', type=float, help='meta-level outer learning rate',
                 default=[1e-4])
        self.add('--modulation-lr', nargs='+', type=float, help='meta-level outer learning rate',
                 default=[1e-4])
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[0.003])
        self.add('--update-step', type=int, help='task-level inner update steps', default=5)
        self.add('--name', help='Name of experiment', default="oml_regression")
        self.add('--model', help='Name of model', default="representation")
        self.add('--model-path', help='Path to model', default=None)
        self.add("--second-order", action="store_true", default=True)
        self.add("--static-plasticity", action="store_true")
        self.add("--no-sigmoid", action="store_true")
        self.add("--neuro", action="store_true")

        self.add("--no-save", action="store_true")
        self.add("--context-plasticity", action="store_true")
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add("--width", type=int, default=400)

