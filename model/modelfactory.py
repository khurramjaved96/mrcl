class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):

        if "Sin" == dataset:
            # if model_type=="old":
            #     hidden_size = width
            #     return [
            #
            #         ('linear', False, [hidden_size, in_channels]),
            #         # ('bn', [hidden_size]),
            #         ('relu', True, [True]),
            #         ('linear', True, [hidden_size, hidden_size]),
            #         # ('bn', [hidden_size]),
            #         ('relu', True, [True]),
            #
            #
            #         ('linear', False, [hidden_size, hidden_size]),
            #         ('relu', True, [True]),
            #         ('linear', True, [hidden_size, hidden_size]),
            #         ('relu', True, [True]),
            #         ('linear', True, [num_actions, hidden_size])
            #     ]
            #
            if model_type == "old":
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
                    ('linear', True, [hidden_size, hidden_size]),
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

            if model_type == "anml":
                hidden_size = width
                return [

                    ('linear', True, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    # ('rep', []),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, [True]),

                    ('modulate', False, None),
                    ('linear', True, [hidden_size, hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [hidden_size, hidden_size]),
                    ('relu', True, [True]),
                    ('linear', True, [num_actions, hidden_size])
                ]

            if model_type == "mrcl":
                hidden_size = width
                return [

                    ('linear', False, True, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    # ('rep', []),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),

                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, False, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, False, [True]),
                    ('linear', True, False, [num_actions, hidden_size])
                ]

            if model_type == "mrcl-all":
                hidden_size = width
                return [

                    ('linear', False, True, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    # ('rep', []),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', False, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('relu', True, False, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('relu', True, False, [True]),
                    ('linear', True, True, [num_actions, hidden_size])
                ]

            if model_type == "maml-warped":
                hidden_size = width
                return [

                    ('linear', True, True, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    # ('rep', []),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', False, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),

                    ('linear', False, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', False, True, [num_actions, hidden_size]),

                ]
            #
            if model_type == "maml":
                hidden_size = width
                return [

                    ('linear', True, True, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    # ('rep', []),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [num_actions, hidden_size]),

                ]

            if model_type == "maml-attention":
                hidden_size = int(width / 4)
                return [
                    #
                    ('positional-attention', True, True, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('positional-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [num_actions, hidden_size]),
                    #
                ]
            #
            if model_type == "maml-mod":
                hidden_size = width
                return [

                    # ('linear-plasticity-bone', False, True, [hidden_size, in_channels]),
                    # ('relu-plasticity-bone', False, True, [True]),
                    # ('linear-plasticity-bone', False, True, [hidden_size, hidden_size]),
                    # ('relu-plasticity-bone', False, True, [True]),
                    #
                    # ('linear-neuromodulation-bone', False, True, [hidden_size, in_channels]),
                    # ('relu-neuromodulation-bone', False, True, [True]),
                    # ('linear-neuromodulation-bone', False, True, [hidden_size, hidden_size]),
                    # ('relu-neuromodulation-bone', False, True, [True]),

                    ('linear', True, True, [hidden_size, in_channels]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [hidden_size, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels]),
                    ('relu', True, True, [True]),

                    ('linear', True, True, [num_actions, hidden_size]),
                    ('modulate', False, False, [hidden_size, in_channels])
                ]

            if model_type == "plasticity":
                hidden_size = width
                return [

                    ('linear', True, False, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    # ('rep', []),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),

                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, False, [num_actions, hidden_size])
                ]

            if model_type == "plasticity-mod":
                hidden_size = width
                return [

                    ('linear', True, False, [hidden_size, in_channels]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    # ('rep', []),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    # ('bn', [hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('modulate', True, False, [hidden_size, in_channels]),
                    ('linear', True, False, [num_actions, hidden_size])
                ]

        elif dataset == "atari":
            width = 300
            return [
                ('conv2d', False, [32, in_channels, 8, 8, 4, 0]),
                ('relu', False, [True]),
                ('conv2d', False, [64, 32, 4, 4, 2, 0]),
                ('relu', False, [True]),
                ('conv2d', False, [64, 64, 3, 3, 1, 0]),
                ('relu', False, [True]),
                ('flatten', True, []),
                ('rep', True, []),
                ('linear', True, [512, 3136]),
                ('relu', False, [True]),
                ('linear', True, [512, 512]),
                ('relu', False, [True]),
                ('linear', True, [num_actions, 512])
            ]

        if dataset == "omniglot-er":
            channels = 128
            # channels = 256
            return [
                ('conv2d', [int(channels / 8), 1, 3, 3, 2, 0]),
                ('relu', [True]),
                ('conv2d', [int(channels / 4), int(channels / 8), 3, 3, 1, 0]),
                ('relu', [True]),
                ('conv2d', [int(channels / 2), int(channels / 4), 3, 3, 2, 0]),
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
                ('conv2d', [int(channels / 8), 1, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('gate', [True]),
                ('bn', [16]),
                ('conv2d', [int(channels / 4), int(channels / 8), 3, 3, 1, 0]),
                ('relu', [True]),
                # ('gate', [True]),
                ('bn', [32]),
                ('conv2d', [int(channels / 2), int(channels / 4), 3, 3, 2, 0]),
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



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
