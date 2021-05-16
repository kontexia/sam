#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
from src.sparse_distributed_representation import SDR
from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder
from src.sam_fabric import SAMFabric
from src.sparse_am_viz import plot_pors, plot_sam


from sklearn.datasets import make_moons, make_swiss_roll


def moon_test():
    data_set, labels = make_moons(n_samples=200,
                                  noise=0.01,
                                  random_state=0)

    training_graphs = []

    numeric_encoder = NumericEncoder(min_step=0.005,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

    for idx in range(len(data_set)):

        xy_sdr = SDR()
        xy_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)
        xy_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)
        label_sdr = SDR()
        label_sdr.add_encoding(enc_key=('label',), value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append({'xy': xy_sdr, 'label': label_sdr})

    sam_params = {'similarity_threshold': 0.7,
                  'community_factor': 0.9,
                  'temporal_learning_rate': 1.0,
                  'prune_threshold': 0.01,
                  'activation_enc_keys': None,
                  'association_learn_rate': 0.6}

    sam_fabric = SAMFabric(association_params=sam_params)

    region_params = {'xy': sam_params,
                     'label': sam_params}

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam_fabric.train(region_sdrs=training_graphs[t_idx], region_sam_params=region_params)
        pors.append(por)

    sam_dict = sam_fabric.to_dict(decode=True)

    plot_sam(sam_region=sam_dict['xy'],
             raw_data=data_set,
             xyz_types=[('x',), ('y',)],
             colour_nodes=None,
             title='Moons')

    xy_pors = [por['xy'] for por in pors]

    plot_pors(xy_pors)

    print('Finished')


def swiss_roll_test():
    data_set, labels = make_swiss_roll(n_samples=200,
                                       noise=0.01,
                                       random_state=0)

    training_graphs = []

    numeric_encoder = NumericEncoder(min_step=0.2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

    for idx in range(len(data_set)):

        xyz_sdr = SDR()
        xyz_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)
        xyz_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)
        xyz_sdr.add_encoding(enc_key=('z',), value=data_set[idx][2], encoder=numeric_encoder)

        training_graphs.append({'xyz': xyz_sdr})

    sam_params = {'similarity_threshold': 0.7,
                  'community_factor': 0.9,
                  'temporal_learning_rate': 1.0,
                  'prune_threshold': 0.01,
                  'activation_enc_keys': None,
                  'association_learn_rate': 0.6}

    sam_fabric = SAMFabric(association_params=sam_params)

    region_params = {'xyz': sam_params}

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam_fabric.train(region_sdrs=training_graphs[t_idx], region_sam_params=region_params)
        pors.append(por)

    sam_dict = sam_fabric.to_dict(decode=True)

    plot_sam(sam_region=sam_dict['xyz'],
             raw_data=data_set,
             xyz_types=[('x',), ('y',), ('z',)],
             colour_nodes=None,
             title='Swiss Roll')

    print('Finished')


def colours():
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

    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    numeric_encoder = NumericEncoder(min_step=2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

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

    n_cycles = 1

    sam_params = {'similarity_threshold': 0.7,
                  'community_factor': 0.9,
                  'temporal_learning_rate': 1.0,
                  'prune_threshold': 0.01,
                  'activation_enc_keys': None,
                  'association_learn_rate': 0.6}

    region_params = {'rgb': sam_params,
                     'label': sam_params}

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

        query_pors = sam_fabric.query(region_sdr={'label': training_graphs[client][10][1]['label']})

        print('finished')


if __name__ == '__main__':

    moon_test()
    swiss_roll_test()
    colours()
