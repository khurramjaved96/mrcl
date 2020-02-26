import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300):

        if "Sin" == dataset:

            if model_type == "mrcl":
                hidden_size = width
                return [

                    ('linear', False, True, [hidden_size, input_dimension]),
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
                    ('linear', True, True, [output_dimension, hidden_size])
                ]

            if model_type == "maml-warped":
                hidden_size = width
                return [

                    ('linear', True, True, [hidden_size, input_dimension]),
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
                    ('linear', False, True, [output_dimension, hidden_size]),

                ]
            #
            if model_type == "context-backbone":
                hidden_size = width
                return [
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

            if model_type == "maml":

                hidden_size = width
                spread = int(np.sqrt(width))
                return [

                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

            if model_type == "maml-sparse":

                hidden_size = width
                # spread = int(np.sqrt(width))
                spread = 20
                return [

                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear-sparse', "adaptation": True, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size, "spread": spread}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

            if model_type == "multihead":
                h = 2
                value_dimension = 5
                total_values = 25
                attention_span = 4
                query_dimension = 15
                init_query_dimension = input_dimension
                return [
                    ('multihead', True, True,
                     [input_dimension, h, query_dimension, value_dimension, total_values, attention_span,
                      init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),

                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [output_dimension, value_dimension * total_values * h]),
                ]

            if model_type == "multihead-mrcl":
                h = 2
                value_dimension = 5
                total_values = 25
                attention_span = 4
                query_dimension = 15
                init_query_dimension = input_dimension
                return [
                    ('multihead', False, True,
                     [input_dimension, h, query_dimension, value_dimension, total_values, attention_span,
                      init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),

                    ('multihead', False, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('linear', True, True,
                     [value_dimension * total_values * h, value_dimension * total_values * h]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [output_dimension, value_dimension * total_values * h]),
                ]

            if model_type == "multihead-3":
                h = 2
                value_dimension = 5
                total_values = 25
                attention_span = 4
                query_dimension = 15
                init_query_dimension = input_dimension
                return [
                    ('multihead-3', True, True,
                     [input_dimension, h, query_dimension, value_dimension, total_values, attention_span,
                      init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),

                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('multihead-3', True, True,
                     [value_dimension * total_values * h, h, query_dimension, value_dimension, total_values,
                      attention_span, init_query_dimension]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [output_dimension, value_dimension * total_values * h]),
                ]

            if model_type == "maml-attention":
                hidden_size = int(width / 4)
                return [
                    #
                    ('positional-attention', True, True, [hidden_size, input_dimension]),
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
                    ('linear', True, True, [output_dimension, hidden_size]),
                    #
                ]

            if model_type == "maml-attention-2":
                hidden_size = int(width / 4)
                return [
                    #
                    ('individual-attention', True, True, [hidden_size, input_dimension]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),

                    ('individual-attention', True, True, [hidden_size, hidden_size]),
                    ('relu', True, True, [True]),
                    ('linear', True, True, [output_dimension, hidden_size]),
                    #
                ]

            if model_type == "plasticity":
                hidden_size = width
                return [

                    ('linear', True, False, [hidden_size, input_dimension]),
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
                    ('linear', True, False, [output_dimension, hidden_size])
                ]


        elif dataset == "atari":
            width = 300
            return [
                ('conv2d', False, [32, input_dimension, 8, 8, 4, 0]),
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
                ('linear', True, [output_dimension, 512])
            ]

        if dataset == "omniglot":
            channels = 128
            # channels = 256
            return [
                ('conv2d', False, True, [channels, 1, 3, 3, 2, 0]),
                ('relu', True, True, [True]),
                # ('bn', [64]),
                ('conv2d', False, True, [channels, channels, 3, 3, 1, 0]),
                ('relu', True, True, [True]),

                ('conv2d', False, True, [channels, channels, 3, 3, 2, 0]),
                ('relu', True, True, [True]),

                ('conv2d', False, True, [channels, channels, 3, 3, 1, 0]),
                ('relu', True, True, [True]),
                # ('bn', [128]),
                ('conv2d', False, True, [channels, channels, 3, 3, 2, 0]),
                ('relu', True, True, [True]),
                # ('bn', [256]),
                ('conv2d', False, True, [channels, channels, 3, 3, 2, 0]),
                ('relu', True, True, [True]),
                # ('bn', [512]),
                ('flatten', True, True, []),

                ('linear', True, True, [1024, 9 * channels]),
                ('relu', True, True, [True]),
                ('linear', True, True, [1000, 1024])
            ]



        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
