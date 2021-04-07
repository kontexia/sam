#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sam import SAM
from src.sdr import SDR
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder
from src.sam_viz import plot_sam, plot_pors

from sklearn.datasets import make_moons, make_swiss_roll
import json


def moon_test():
    data_set, labels = make_moons(n_samples=200,
                                  noise=0.01,
                                  random_state=0)

    training_graphs = []

    numeric_encoder = NumericEncoder(min_step=0.005,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = StringEncoder(n_bits=40,
                                  enc_size=2048,
                                  seed=456)

    for idx in range(len(data_set)):

        t_sdr = SDR()
        t_sdr.add_encoding('x', value=data_set[idx][0], encoder=numeric_encoder)
        t_sdr.add_encoding('y', value=data_set[idx][1], encoder=numeric_encoder)
        t_sdr.add_encoding('label', value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append(t_sdr)

    sam = SAM(name='MoonTest',
              similarity_threshold=0.75,
              anomaly_threshold_factor=3.0,
              similarity_decay=0.1,
              learn_rate_decay=0.3,
              prune_threshold=0.01,
              prune_neurons=False)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam.train(sdr=training_graphs[t_idx],
                        ref_id=str(t_idx),
                        search_types={'x', 'y', 'label'},
                        learn_types={'x', 'y', 'label'})
        pors.append(por)

    sam_dict = sam.to_dict(decode=True)

    plot_sam(sam=sam_dict,
             raw_data=data_set,
             xyz_types=['x', 'y'],
             colour_nodes='label')

    plot_pors(pors=pors)

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

    label_encoder = StringEncoder(n_bits=40,
                                  enc_size=2048,
                                  seed=456)

    for idx in range(len(data_set)):

        t_sdr = SDR()
        t_sdr.add_encoding('x', value=data_set[idx][0], encoder=numeric_encoder)
        t_sdr.add_encoding('y', value=data_set[idx][1], encoder=numeric_encoder)
        t_sdr.add_encoding('z', value=data_set[idx][2], encoder=numeric_encoder)

        t_sdr.add_encoding('label', value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append(t_sdr)

    sam = SAM(name='SwissTest',
              similarity_threshold=0.75,
              anomaly_threshold_factor=3.0,
              similarity_decay=0.1,
              learn_rate_decay=0.3,
              prune_threshold=0.01,
              prune_neurons=False)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam.train(sdr=training_graphs[t_idx],
                        ref_id=str(t_idx),
                        search_types={'x', 'y', 'z'},
                        learn_types={'x', 'y', 'z'})
        pors.append(por)

    sam_dict = sam.to_dict(decode=True)

    plot_sam(sam=sam_dict,
             raw_data=data_set,
             xyz_types=['x', 'y', 'z'],
             colour_nodes=None)

    plot_pors(pors=pors)

    print('Finished')


def colours_test():
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

    numeric_encoder = NumericEncoder(min_step=1,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = StringEncoder(n_bits=40,
                                  enc_size=2048,
                                  seed=456)

    training_graphs = {}
    for record in raw_data:
        if record['client'] not in training_graphs:
            training_graphs[record['client']] = []

        t_sdr = SDR()
        t_sdr.add_encoding('r', value=record['r'], encoder=numeric_encoder)
        t_sdr.add_encoding('g', value=record['g'], encoder=numeric_encoder)
        t_sdr.add_encoding('b', value=record['b'], encoder=numeric_encoder)
        t_sdr.add_encoding('label', value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b']]

        training_graphs[record['client']].append((record['trade_id'], t_sdr, r_data))

    n_cycles = 1

    sams = {}
    for client in training_graphs:
        pors = []
        sam = SAM(name=client,
                  similarity_threshold=0.75,
                  anomaly_threshold_factor=3.0,
                  similarity_decay=0.1,
                  learn_rate_decay=0.3,
                  prune_threshold=0.01,
                  prune_neurons=False)

        sams[client] = {'sam': sam}

        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = sam.train(sdr=training_graphs[client][t_idx][1],
                                ref_id=str(t_idx),
                                search_types={'r', 'g', 'b', 'colour'},
                                learn_types={'r', 'g', 'b', 'colour'})
                pors.append(por)

        sams[client]['por'] = pors

        #ng.calc_communities()

        plot_pors(pors=pors)

        sam_dict = sam.to_dict(decode=True)

        rn_data = [t_data[2] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam=sam_dict,
                 raw_data=cycle_data,
                 xyz_types=['r', 'g', 'b'],
                 colour_nodes=None)

        print(f'finished {client}')




if __name__ == '__main__':

    colours_test()
    #moon_test()
    #swiss_roll_test()
