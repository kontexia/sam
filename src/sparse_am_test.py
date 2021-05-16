#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import json
from src.sparse_associative_memory import SAM
from src.sparse_distributed_representation import SDR
from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder
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

        t_sdr = SDR()
        t_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('label',), value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append(t_sdr)

    sam = SAM(name='Moon',
              similarity_threshold=0.7,
              community_factor=0.8,
              temporal_learn_rate=1.0)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam.learn_pattern(sdr=training_graphs[t_idx],
                                activation_enc_keys={('x',), ('y',)})
        pors.append(por)

    sam_dict = sam.to_dict(decode=True)

    plot_sam(sam_region=sam_dict,
             raw_data=data_set,
             xyz_types=[('x',), ('y',)],
             colour_nodes=None,
             title='Moons')

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

        t_sdr = SDR()
        t_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('z',), value=data_set[idx][2], encoder=numeric_encoder)

        t_sdr.add_encoding(enc_key=('label',), value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append(t_sdr)

    sam = SAM(name='Swiss',
              similarity_threshold=0.85,
              community_factor=0.9,
              temporal_learn_rate=1.0)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam.learn_pattern(sdr=training_graphs[t_idx],
                                activation_enc_keys={('x',), ('y',), ('z',)})
        pors.append(por)

    sam_dict = sam.to_dict(decode=True)

    plot_sam(sam_region=sam_dict,
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

        t_sdr = SDR()
        t_sdr.add_encoding(enc_key=('r',), value=record['r'], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('g',), value=record['g'], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('b',), value=record['b'], encoder=numeric_encoder)
        t_sdr.add_encoding(enc_key=('label',), value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b'], record['label']]

        training_graphs[record['client']].append((record['trade_id'], t_sdr, r_data))

    n_cycles = 1

    sams = {}
    for client in training_graphs:
        pors = []
        sam = SAM(name=client,
                  similarity_threshold=0.7,
                  community_factor=0.9,
                  temporal_learn_rate=0.6)

        sams[client] = {'sam': sam}

        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = sam.learn_pattern(sdr=training_graphs[client][t_idx][1],
                                        activation_enc_keys={('r',), ('g',), ('b',)})
                pors.append(por)

        sams[client]['por'] = pors

        plot_pors(pors, title=client)

        sam_dict = sam.to_dict(decode=True)

        rn_data = [t_data[2][:3] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam_region=sam_dict,
                 raw_data=cycle_data,
                 xyz_types=[('r',), ('g',), ('b',)],
                 colour_nodes=None,
                 temporal_key=0,
                 title=client)

        print(f'finished {client}')


if __name__ == '__main__':

    moon_test()
    swiss_roll_test()
    colours()
