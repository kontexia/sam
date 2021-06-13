#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_distributed_representation import SDR
from src.sam_fabric import SAM
from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder
from time import time
import random


def performance_test():
    regional_sam_params = {
        # an incoming training sdr must be at least 70% similar to a neuron to be mapped to it
        'similarity_threshold': 0.7,

        # neurons that are at least 63% (0.7 * 0.9) similar to the incoming sdr are considered to be in the same community
        'community_factor': 0.9,

        # the level below which a weight is considered zero and will be deleted
        'prune_threshold': 0.01,

        # a set of enc_type tuples to be used in learning - setting to None implies all enc_types will be learned
        'activation_enc_keys': None}

    association_sam_params = {
        'similarity_threshold': 0.6,
        'community_factor': 0.9,
        'prune_threshold': 0.01}

    temporal_sam_params = {
        'similarity_threshold': 0.6,
        'community_factor': 0.9,
        'prune_threshold': 0.01,
        'lstm_len': 4}

    # we have the possibility of having different configurations per region
    #
    region_params = {'rgb': regional_sam_params,
                     'label': regional_sam_params}

    sam_fabric = SAM(spatial_params=region_params, association_params=association_sam_params, temporal_params=temporal_sam_params)

    # create a numeric encoder and label encoder
    #
    numeric_encoder = NumericEncoder(min_step=2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

    n_epochs = 1000
    n_attributes = 10
    n_labels = 3
    label = 0
    random.seed(123)
    rand_state = random.getstate()
    pors = []
    for epoch in range(n_epochs):

        rgb_sdr = SDR()
        for attribute_idx in range(n_attributes):
            random.setstate(rand_state)
            value = random.random()*255
            rand_state = random.getstate()

            rgb_sdr.add_encoding(enc_key=(str(attribute_idx),), value=value, encoder=numeric_encoder)

        label_sdr = SDR()
        label_sdr.add_encoding(enc_key=('label',), value=str(label), encoder=label_encoder)
        label += 1
        if label > n_labels:
            label = 0

        por = sam_fabric.train(sdrs={'rgb': rgb_sdr, 'label': label_sdr})
        pors.append(por)

    timings = {'rgb': {'total_time': 0.0,
                       'avg_time': 0.0,
                       'activate_neuron_total': 0.0,
                       'avg_activate_neuron': 0.0,
                       'nos_activate_neuron': 1,

                       'create_neuron_total': 0.0,
                       'avg_create_neuron': 0.0,
                       'nos_create_neuron': 1,

                       'update_neuron_total': 0.0,
                       'avg_update_neuron': 0.0,
                       'nos_update_neuron': 1,

                       'update_community_total': 0.0,
                       'avg_update_community': 0.0,
                       },

               'label': {'total_time': 0.0,
                         'avg_time': 0.0,
                         'activate_neuron_total': 0.0,
                         'avg_activate_neuron': 0.0,
                         'nos_activate_neuron': 1,

                         'create_neuron_total': 0.0,
                         'avg_create_neuron': 0.0,
                         'nos_create_neuron': 1,

                         'update_neuron_total': 0.0,
                         'avg_update_neuron': 0.0,
                         'nos_update_neuron': 1,

                         'update_community_total': 0.0,
                         'avg_update_community': 0.0,
                         },

               '_association': {'total_time': 0.0,
                                'avg_time': 0.0,
                                'activate_neuron_total': 0.0,
                                'avg_activate_neuron': 0.0,
                                'nos_activate_neuron': 1,

                                'create_neuron_total': 0.0,
                                'avg_create_neuron': 0.0,
                                'nos_create_neuron': 1,

                                'update_neuron_total': 0.0,
                                'avg_update_neuron': 0.0,
                                'nos_update_neuron': 1,

                                'update_community_total': 0.0,
                                'avg_update_community': 0.0,
                                },

               '_temporal': {'total_time': 0.0,
                             'avg_time': 0.0,
                             'activate_neuron_total': 0.0,
                             'nos_activate_neuron': 1,
                             'avg_activate_neuron': 0.0,
                             'create_neuron_total': 0.0,
                             'avg_create_neuron': 0.0,
                             'nos_create_neuron': 1,
                             'update_neuron_total': 0.0,
                             'avg_update_neuron': 0.0,
                             'nos_update_neuron': 1,
                             'update_community_total': 0.0,
                             'avg_update_community': 0.0,
                             }
               }
    for por in pors:
        for pooler in por:

            if 'learn_sec' in por[pooler]:
                timings[pooler]['total_time'] += por[pooler]['learn_sec']
                timings[pooler]['avg_time'] += por[pooler]['learn_sec'] / n_epochs

                if por[pooler]['activate_neuron_sec'] is not None:
                    timings[pooler]['activate_neuron_total'] += por[pooler]['activate_neuron_sec']
                    timings[pooler]['nos_activate_neuron'] += 1

                if por[pooler]['create_neuron_sec'] is not None:
                    timings[pooler]['create_neuron_total'] += por[pooler]['create_neuron_sec']
                    timings[pooler]['nos_create_neuron'] += 1

                if por[pooler]['update_neuron_sec'] is not None:
                    timings[pooler]['update_neuron_total'] += por[pooler]['update_neuron_sec']
                    timings[pooler]['nos_update_neuron'] += 1

                timings[pooler]['update_community_total'] += por[pooler]['update_community_sec']
                timings[pooler]['avg_update_community'] += por[pooler]['update_community_sec'] / n_epochs

    for pooler in timings:
        print(pooler)
        print(f"  total time: {timings[pooler]['total_time']} avg: {timings[pooler]['avg_time']} nos: {n_epochs}")
        print(f"  activate time: {timings[pooler]['activate_neuron_total']} avg: {timings[pooler]['activate_neuron_total'] / timings[pooler]['nos_activate_neuron']} nos: {timings[pooler]['nos_activate_neuron']}")
        print(f"  create time: {timings[pooler]['create_neuron_total']} avg: {timings[pooler]['create_neuron_total'] / timings[pooler]['nos_create_neuron']} nos: {timings[pooler]['nos_create_neuron']}")
        print(f"  update time: {timings[pooler]['update_neuron_total']} avg: {timings[pooler]['update_neuron_total'] / timings[pooler]['nos_update_neuron']} nos: {timings[pooler]['nos_update_neuron']}")
        print(f"  community time: {timings[pooler]['update_community_total']} avg: {timings[pooler]['update_community_total']} nos: {n_epochs}")


if __name__ == '__main__':
    performance_test()
