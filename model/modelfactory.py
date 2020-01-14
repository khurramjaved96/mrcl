class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):

        #
        if "Sin" == dataset:
            if model_type=="old":
                hidden_size = width
                return [

                    ('linear', False, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    # ('rep', []),
                    ('linear', False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True,  [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),

                    ('linear', False, [hidden_size, hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [num_actions, hidden_size])
                ]
        #
        if dataset == "omniglot-er":
            channels = 128
            # channels = 256
            return [
                ('conv2d', [int(channels/8), 1, 3, 3, 2, 0]),
                ('relu', [True]),
                ('conv2d', [int(channels/4), int(channels/8), 3, 3, 1, 0]),
                ('relu', [True]),
                ('conv2d', [int(channels/2), int(channels/4), 3, 3, 2, 0]),
                ('relu', [True]),
                ('flatten', []),
                ('rep', []),
                ('linear', [1024, 18 * channels]),
                ('relu', [True]),
                ('linear', [1000, 1024])
            ]

        if dataset == "omniglot":
            channels = 128
            # channels = 256
            return [
                ('conv2d', [int(channels/8), 1, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('gate', [True]),
                ('bn', [16]),
                ('conv2d', [int(channels/4), int(channels/8), 3, 3, 1, 0]),
                ('relu', [True]),
                # ('gate', [True]),
                ('bn', [32]),
                ('conv2d', [int(channels/2), int(channels/4), 3, 3, 2, 0]),
                ('relu', [True]),
                # ('gate', [True]),
                ('bn', [64]),
                ('flatten', []),
                ('rep', []),
                ('gate', [True]),
                ('linear', [1024, 18 * channels]),
                ('relu', [True]),
                # ('gate', [True]),

                ('linear', [1000, 1024])
            ]

        # if dataset == "omniglot":
        #     channels = 32
        #     # channels = 256
        #     return [
        #         ('conv2d', [int(channels/2), 1, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [64]),
        #         ('conv2d', [int(channels/2), int(channels/2), 3, 3, 1, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [int(channels/2), int(channels/2), 3, 3, 2, 0]),
        #         ('relu', [True]),
        #
        #         ('conv2d', [channels, int(channels/2), 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [128]),
        #         ('conv2d', [channels, channels, 3, 3, 1, 0]),
        #         ('relu', [True]),
        #         # ('bn', [256]),
        #         ('conv2d', [channels, channels, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [512]),
        #         ('flatten', []),
        #         ('rep', []),
        #
        #         ('linear', [256, 9 * channels]),
        #         ('relu', [True]),
        #         ('linear', [1000, 256])
        #     ]

        if dataset == "omniglot-pln":
            channels = 256
            # channels = 256
            return [
                ('linear', [1024, 9 * channels]),
                ('relu', [True]),
                ('linear', [1000, 1024])
            ]
        # if dataset == 'celeba':
        #     channels = 256
        #     return [
        #         ('conv2d', [int(channels/2), 3, 3, 3, 2, 0]),
        #         ('relu', [True]),
        #         # ('bn', [64]),
        #         ('conv2d', [channels, int(channels/2), 3, 3, 1, 0]),
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
        #         ('linear', [11000, 1024])
        #     ]
        #
        if dataset == 'celeba':
            channels = 256
            return [
                ('conv2d', [int(channels/8), 3, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [int(channels/4), int(channels/8), 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [int(channels/4), int(channels/4), 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [int(channels/2), int(channels/4), 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, int(channels/2), 3, 3, 1, 1]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [256,  channels]),
                ('relu', [True]),
                ('linear', [11000, 256])
            ]

        if dataset == 'imagenet':
            channels = 64
            # channels = 256
            return [
                ('conv2d', [channels, 3, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [512, 9 * channels]),
                ('relu', [True]),
                ('linear', [100, 512])
            ]

        if dataset == 'fullimagenet':
            const =1
            # channels = 256
            return [
                ('conv2d', [64, 3, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('max_pool2d', [2, 2, 0]),

                ('conv2d', [128, 64, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('max_pool2d', [2, 2, 0]),

                ('conv2d', [256, 128, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('max_pool2d', [2, 2, 0]),

                ('conv2d', [512, 256, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('max_pool2d', [2, 2, 0]),

                ('conv2d', [512, 512, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),
                ('relu', [True]),
                ('gate', [True]),
                ('max_pool2d', [2, 2, 0]),

                ('flatten', []),
                ('rep', []),

                ('linear', [4096, 25088]),
                ('relu', [True]),
                ('gate', [True]),
                # ('linear', [4096, 25088]),
                # ('relu', [True]),
                ('linear', [1000, 4096])
            ]



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
