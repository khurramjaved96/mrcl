import logging

logger = logging.getLogger("experiment")

import torch
from torch import nn


def initialize_context_plasticity(param1, param2):
    w_context_plasticity = nn.Parameter(
        torch.ones(param1, param2))
    torch.nn.init.kaiming_normal_(w_context_plasticity)
    b_context_plasticity = nn.Parameter(torch.zeros(param1))
    w_context_plasticity.learn = False
    b_context_plasticity.learn = False
    w_context_plasticity.meta = True
    b_context_plasticity.meta = True
    return w_context_plasticity, b_context_plasticity


def initialize_plasticity(param1, param2):
    w = nn.Parameter(torch.zeros(param1, param2))
    b = nn.Parameter(torch.zeros(param1))
    w.learn = False
    b.learn = False
    w.meta = True
    b.meta = True
    return w, b


def initialize_attention(param1, param2, query_span, attention_span, adaptation, meta):
    w_values = nn.Parameter(torch.ones(param1 * attention_span, param2))
    w_keys = nn.Parameter(torch.ones(query_span * attention_span, param2))

    w_values_bias = nn.Parameter(torch.zeros(param1 * attention_span))
    w_keys_bias = nn.Parameter(torch.zeros(query_span * attention_span))

    w_values.learn = adaptation
    w_values_bias.learn = adaptation

    w_values.meta = meta
    w_values_bias.meta = meta

    w_keys.learn = False
    w_keys_bias.learn = False

    w_keys.meta = True
    w_keys_bias.meta = True

    torch.nn.init.kaiming_normal_(w_values)
    torch.nn.init.kaiming_normal_(w_keys)
    return w_values, w_values_bias, w_keys, w_keys_bias


def initialize_positional_attention(param1, param2, query_span, attention_span, adaptation, meta):
    w_values = nn.Parameter(torch.ones(param1 * attention_span, param2))
    w_values_bias = nn.Parameter(torch.zeros(param1 * attention_span))

    w_query = nn.Parameter(torch.ones(attention_span, query_span))

    w_query_bias = nn.Parameter(torch.zeros(attention_span))

    w_values.learn = adaptation
    w_values_bias.learn = adaptation

    w_values.meta = meta
    w_values_bias.meta = meta

    w_query.learn = False
    w_query_bias.learn = False

    w_query.meta = True
    w_query_bias.meta = True

    torch.nn.init.kaiming_normal_(w_values)
    torch.nn.init.kaiming_normal_(w_query)
    return w_values, w_values_bias, w_query, w_query_bias


def initialize_individual_attention(param1, param2, query_span, attention_span, adaptation, meta):
    w_values = nn.Parameter(torch.ones(param1 * attention_span, param2))
    w_values_bias = nn.Parameter(torch.zeros(param1 * attention_span))

    w_query = nn.Parameter(torch.ones(param1 * attention_span, query_span))

    w_query_bias = nn.Parameter(torch.zeros(param1 * attention_span))

    w_values.learn = adaptation
    w_values_bias.learn = adaptation

    w_values.meta = meta
    w_values_bias.meta = meta

    w_query.learn = False
    w_query_bias.learn = False

    w_query.meta = True
    w_query_bias.meta = True

    torch.nn.init.kaiming_normal_(w_values)
    torch.nn.init.kaiming_normal_(w_query)
    return w_values, w_values_bias, w_query, w_query_bias


def initialize_multihead(embedding_size, h, query_dimension, value_dimension, total_values, attention_span,
                         init_query_dimension, adaptation, meta):
    w_queries = nn.Parameter(torch.ones(query_dimension * h, init_query_dimension))
    w_keys = nn.Parameter(torch.ones(query_dimension * total_values * attention_span, embedding_size))
    w_values = nn.Parameter(torch.ones(value_dimension * total_values * attention_span, embedding_size))
    w_values_bias = nn.Parameter(torch.zeros(value_dimension * total_values * attention_span))

    w_values.learn = adaptation
    w_values_bias.learn = adaptation

    w_values.meta = meta
    w_values_bias.meta = meta

    w_queries.learn = False
    w_keys.learn = False

    w_keys.meta = True
    w_queries.meta = True

    torch.nn.init.kaiming_normal_(w_values)
    torch.nn.init.kaiming_normal_(w_queries)
    torch.nn.init.kaiming_normal_(w_keys)

    return w_values, w_values_bias, w_queries, w_keys


def initialize_multihead_3(embedding_size, h, query_dimension, value_dimension, total_values, attention_span,
                           init_query_dimension, adaptation, meta):
    w_queries = nn.Parameter(torch.ones(query_dimension * h, init_query_dimension))
    w_keys = nn.Parameter(torch.ones(query_dimension * total_values * attention_span, embedding_size))
    w_values = nn.Parameter(torch.ones(value_dimension * total_values * attention_span, embedding_size))
    w_values_bias = nn.Parameter(torch.zeros(value_dimension * total_values * attention_span))

    w_values.learn = adaptation
    w_values_bias.learn = adaptation

    w_values.meta = meta
    w_values_bias.meta = meta

    w_queries.learn = adaptation
    w_keys.learn = adaptation

    w_keys.meta = meta
    w_queries.meta = meta

    torch.nn.init.kaiming_normal_(w_values)
    torch.nn.init.kaiming_normal_(w_queries)
    torch.nn.init.kaiming_normal_(w_keys)

    return w_values, w_values_bias, w_queries, w_keys


#
def initialize_linear(param1, param2, adaptation, meta):
    w = nn.Parameter(torch.ones(param1, param2))
    w.learn = adaptation
    w.meta = meta
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(param1))
    b.learn = adaptation
    b.meta = meta
    return w, b
