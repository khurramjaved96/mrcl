class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):

        if "Sin" == dataset:
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



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
