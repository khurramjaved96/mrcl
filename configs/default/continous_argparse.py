import configargparse


class DefaultParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('-c', '--my-config', is_config_file=True, default="configs/default/default.ini",
                 help='config file path')
        self.add('--epoch', nargs='+', type=int)
        self.add('--seed', nargs='+', help='Seed', default=[20, 30, 40, 50, 60], type=int)
        self.add('--dataset', help='Name of experiment', default="Adversarial_adaptation")
        self.add('--output-dir', help='Output directory', default="../result/")
        self.add('--lr', nargs='+', help='Name of experiment', default=[1e-4, 1e-3, 1e-5], type=float)
        self.add('--name', help='Name of experiment', default="Latent_Adversarial")
        self.add('--batch-size', nargs='+', help='Batch size', default=[1, 4, 16, 64], type=int)

        self.add("--commit", action="store_true")
