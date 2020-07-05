import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300):

        if "Sin" == dataset:

            if model_type == "representation":

                hidden_size = width
                return [

                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

        elif dataset == "omniglot":
            channels = 256

            return [
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}}

            ]

        #
        # elif dataset == "omniglot":
        #     channels = 256
        #     # channels = 256
        #     return [
        #         ('conv2d', [channels, 1, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [64]),
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #         # ('bn', [128]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [256]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [512]),
        #         ('flatten', []),
        #         ('rep', []),
        #
        #         {"name": 'linear', "adaptation": True, "meta": True,
        #          "config": {"out": 1000, "in": 9*channels}}
        #
        #     ]
        #
        # elif dataset == "omniglot-few":
        #     channels = 256
        #     # channels = 256
        #     return [
        #         ('conv2d', [channels, 1, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         ('bn', [channels]),
        #         ('flatten', []),
        #         ('rep', []),
        #
        #         ('linear', [1000, 9 * channels])
        #     ]
        #
        # elif dataset == "omniglot-fc":
        #     channels = 256
        #     # channels = 256
        #     return [
        #         ('flatten', []),
        #         ('linear', [256, 84*84]),
        #         ('relu', [True]),
        #         ('linear', [128, 256]),
        #         ('relu', [True]),
        #         ('linear', [64, 128]),
        #         ('relu', [True]),
        #         ('rep', []),
        #         ('linear', [5, 64]),
        #     ]
        #
        # elif dataset == "imagenet":
        #     channels = 256
        #     # channels = 256
        #     return [
        #         ('conv2d', [channels, 3, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [64]),
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #         # ('bn', [128]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [256]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [512]),
        #         ('flatten', []),
        #         ('rep', []),
        #
        #         ('linear', [1024, 9 * channels]),
        #         ('relu', [True]),
        #         ('linear', [1000, 1024])
        #     ]




        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
