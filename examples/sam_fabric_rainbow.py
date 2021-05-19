#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_distributed_representation import SDR
from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder
from src.sam_fabric import SAMFabric
from src.sparse_am_viz import plot_pors, plot_sam
import json


def rainbow():

    """
    colours = {'RED': {'r': 255, 'b': 0, 'g': 0},
               'ORANGE': {'r': 255, 'b': 129, 'g': 0},
               'YELLOW': {'r': 255, 'b': 233, 'g': 0},
               'GREEN': {'r': 0, 'b': 202, 'g': 14},
               'BLUE': {'r': 22, 'b': 93, 'g': 239},
               'PURPLE': {'r': 166, 'b': 1, 'g': 214},
               'BROWN': {'r': 151, 'b': 76, 'g': 2},
               'GREY': {'r': 128, 'b': 128, 'g': 128},
               'BLACK': {'r': 0, 'b': 0, 'g': 0},
               'TURQUOISE': {'r': 150, 'b': 255, 'g': 255},
               }
    """

    # read in rainbow trades
    #
    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    # create a numeric encoder and label encoder
    #
    numeric_encoder = NumericEncoder(min_step=2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

    # construct a training set of SDRs encoding the rgb values of the trades and
    # encoding the label
    #
    training_graphs = {}
    for record in raw_data:
        if record['client'] not in training_graphs:
            training_graphs[record['client']] = []

        rgb_sdr = SDR()
        rgb_sdr.add_encoding(enc_key=('r',), value=record['r'], encoder=numeric_encoder)
        rgb_sdr.add_encoding(enc_key=('g',), value=record['g'], encoder=numeric_encoder)
        rgb_sdr.add_encoding(enc_key=('b',), value=record['b'], encoder=numeric_encoder)

        label_sdr = SDR()
        label_sdr.add_encoding(enc_key=('label',), value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b'], record['label']]

        training_graphs[record['client']].append((record['trade_id'], {'rgb': rgb_sdr, 'label': label_sdr}, r_data))

    sam_params = {
        # an incoming training sdr must be at least 70% similar to a neuron to be mapped to it
        'similarity_threshold': 0.7,

        # neurons that are at least 63% (0.7 * 0.9) similar to the incoming sdr are considered to be in the same community
        'community_factor': 0.9,

        # setting a temporal_learning_rate to 1.0 effectively turns off temporal learning as 100% is learned from the
        # incoming sdr and 0% (1 - 1.0) is remembered from the previous SDRs
        'temporal_learning_rate': 1.0,

        # the level below which a weight is considered zero and will be deleted
        'prune_threshold': 0.01,

        # a set of enc_type tuples to be used in learning - settin to None implies all enc_types will be learned
        'activation_enc_keys': None,

        # the learning rate of associative connections between neurons of different regions
        'association_learn_rate': 0.6}

    # we have the possibility of having different configurations per region
    #
    region_params = {'rgb': sam_params,
                     'label': sam_params}


    n_cycles = 1
    sams = {}
    for client in training_graphs:
        pors = []
        sam_fabric = SAMFabric(association_params=sam_params)

        sams[client] = {'sam': sam_fabric, 'por': []}

        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = sam_fabric.train(region_sdrs=training_graphs[client][t_idx][1], region_sam_params=region_params)
                pors.append(por)

        sams[client]['por'] = pors

        rgb_pors = [por['rgb'] for por in pors]

        plot_pors(rgb_pors, title=client)

        assoc_pors = [por['association'] for por in pors]

        plot_pors(assoc_pors)

        sam_dict = sam_fabric.to_dict(decode=True)

        rn_data = [t_data[2][:3] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam_region=sam_dict['rgb'],
                 raw_data=cycle_data,
                 xyz_types=[('r',), ('g',), ('b',)],
                 colour_nodes=None,
                 temporal_key=0,
                 title=client)

        # query the sam using only the label to find the associated rainbow trades
        #
        query_pors = sam_fabric.query(region_sdr={'label': training_graphs[client][10][1]['label']})

        print(query_pors)


if __name__ == '__main__':

    rainbow()
