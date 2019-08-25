class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):

        if "Sin" == dataset:
            if model_type=="old":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 3, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 3]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size])
                ]
            elif model_type=="linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [num_actions, hidden_size * 5])
                ]

            elif model_type=="non-linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 5]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size])

                ]

        elif dataset == "omniglot":
            channels = 256
            # channels = 256
            return [
                ('conv2d', [channels, 1, 3, 3, 2, 0]),
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

                ('linear', [1024, 9 * channels]),
                ('relu', [True]),
                ('linear', [1000, 1024])
            ]

        elif dataset == "omniglot-few":
            channels = 256
            # channels = 256
            return [
                ('conv2d', [channels, 1, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [channels]),
                ('flatten', []),
                ('rep', []),

                ('linear', [1024, 9 * channels]),
                ('relu', [True]),
                ('linear', [1000, 1024])
            ]

        elif dataset == "omniglot-fc":
            channels = 256
            # channels = 256
            return [
                ('flatten', []),
                ('linear', [256, 84*84]),
                ('relu', [True]),
                ('linear', [128, 256]),
                ('relu', [True]),
                ('linear', [64, 128]),
                ('relu', [True]),
                ('rep', []),
                ('linear', [5, 64]),
            ]

        elif dataset == "imagenet":
            channels = 256
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

                ('linear', [1024, 9 * channels]),
                ('relu', [True]),
                ('linear', [1000, 1024])
            ]




        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
