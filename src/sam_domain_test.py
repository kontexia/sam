#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
from src.sgm import SGM
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder
from src.sam_domain import SAMDomain

from src.sam_viz import plot_sam, plot_pors


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
            domains[client] = {'fabric': SAMDomain(domain=client,
                                                   spatial_search_types={'r', 'g', 'b'},
                                                   spatial_learn_types={'r', 'g', 'b', 'label'},
                                                   spatial_similarity_threshold=0.75,
                                                   community_similarity_threshold=0.05,
                                                   temporal_similarity_threshold=0.75,
                                                   temporal_memory=10,
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

        plot_pors(pors=pors, name='spatial')
        plot_pors(pors=pors, name='temporal')

        spatial_dict = domains[client]['fabric'].sams['spatial'].to_dict(decode=True)
        temporal_dict = domains[client]['fabric'].sams['temporal'].to_dict(decode=True)

        rn_data = [t_data[2][:3] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam=spatial_dict,
                 raw_data=cycle_data,
                 xyz_types=['r', 'g', 'b'],
                 colour_nodes=None)

        plot_sam(sam=temporal_dict,
                 raw_data=cycle_data,
                 xyz_types=['z0_r', 'z0_g', 'z0_b'],
                 colour_nodes=None)

        start_idx = 58
        por_qry = domains[client]['fabric'].query_temporal(context_sgms=[training_graphs[client][q_idx][1] for q_idx in range(start_idx, start_idx + 2)], bmu_only=False)
        actual_sdr = por_qry['sgm'].to_dict(decode=True)
        target_sdr = training_graphs[client][start_idx + 2][1].to_dict(decode=True)

        print(f'finished {client}')


if __name__ == '__main__':

    colours_test()
