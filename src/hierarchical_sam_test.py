#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
from src.sgm import SGM
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder
from src.hierarchical_sam import HSAM

from src.hierarchical_sam_viz import plot_sam, plot_pors


def colours_test():
    import time
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

    label_encoder = StringEncoder(n_bits=40,
                                  enc_size=2048,
                                  seed=456)

    training_graphs = {}
    for record in raw_data:
        if record['client'] not in training_graphs:
            training_graphs[record['client']] = []

        t_sdr = SGM()
        t_sdr.add_encoding('r', value=record['r'], encoder=numeric_encoder)
        t_sdr.add_encoding('g', value=record['g'], encoder=numeric_encoder)
        t_sdr.add_encoding('b', value=record['b'], encoder=numeric_encoder)
        t_sdr.add_encoding('label', value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b'], record['label']]

        training_graphs[record['client']].append((record['trade_id'], t_sdr, r_data))

    n_cycles = 1

    domains = {}
    for client in training_graphs:
        pors = []
        if client not in domains:
            domains[client] = {'fabric': HSAM(domain=client,
                                              spatial_search_types={'r', 'g', 'b'},
                                              spatial_learn_types={'r', 'g', 'b', 'label'},
                                              spatial_similarity_threshold=0.75,
                                              community_threshold_adj=0.3,
                                              anomaly_threshold_factor=3.0,
                                              similarity_ema_alpha=0.3,
                                              learn_rate_decay=0.3,
                                              prune_threshold=0.01)}

        start_time = time.time()
        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = domains[client]['fabric'].train(sgm=training_graphs[client][t_idx][1],
                                                      ref_id=str(t_idx))
                pors.append(por)
        end_time = time.time()
        print((end_time-start_time) / (n_cycles + len(training_graphs[client])))
        domains[client]['pors'] = pors

        rn_data = [t_data[2][:3] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        for domain in domains[client]['fabric'].sams:
            plot_pors(pors=pors, name=domain)

            spatial_dict = domains[client]['fabric'].sams[domain].to_dict(decode=True)

            plot_sam(sam=spatial_dict,
                     raw_data=cycle_data,
                     xyz_types=['r', 'g', 'b'],
                     colour_nodes=None)

        print(f'finished {client}')


if __name__ == '__main__':

    colours_test()
