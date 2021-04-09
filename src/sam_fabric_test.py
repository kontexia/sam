#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
from src.sgm import SGM
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder
from src.sam_fabric import SAMFabric

from src.sam_viz import plot_sam, plot_pors
import time


def colours_association_test():
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

        r_sdr = SGM(enc_type='r', value=record['r'], encoder=numeric_encoder)
        g_sdr = SGM(enc_type='g', value=record['g'], encoder=numeric_encoder)
        b_sdr = SGM(enc_type='b', value=record['b'], encoder=numeric_encoder)
        l_sdr = SGM(enc_type='label', value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b']]

        training_graphs[record['client']].append((record['trade_id'], [r_sdr, g_sdr, b_sdr, l_sdr], r_data))

    n_cycles = 1

    domains = {}
    for client in training_graphs:
        pors = []
        domains[client] = {'fabric': SAMFabric(domain=client,
                                               assoc_similarity_threshold=0.75,
                                               assoc_anomaly_threshold_factor=3.0,
                                               temporal_similarity_threshold=0.75,
                                               temporal_anomaly_threshold_factor=3.0,
                                               similarity_ema_alpha=0.1,
                                               learn_rate_decay=0.3,
                                               prune_threshold=0.01,
                                               prune_neurons=False,
                                               )}

        domains[client]['fabric'].create_sam(name='r',
                                             similarity_threshold=0.75,
                                             anomaly_threshold_factor=3.0,
                                             search_types={'r'},
                                             learn_types={'r'},
                                             create_temporal=True)
        domains[client]['fabric'].create_sam(name='g',
                                             similarity_threshold=0.75,
                                             anomaly_threshold_factor=3.0,
                                             search_types={'g'},
                                             learn_types={'g'},
                                             create_temporal=True)
        domains[client]['fabric'].create_sam(name='b',
                                             similarity_threshold=0.75,
                                             anomaly_threshold_factor=3.0,
                                             search_types={'b'},
                                             learn_types={'b'},
                                             create_temporal=True)
        domains[client]['fabric'].create_sam(name='l',
                                             similarity_threshold=0.75,
                                             anomaly_threshold_factor=3.0,
                                             search_types={'l'},
                                             learn_types={'l'},
                                             create_temporal=True)

        start_time = time.time()
        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = domains[client]['fabric'].train(sgms={'r': training_graphs[client][t_idx][1][0],
                                                            'g': training_graphs[client][t_idx][1][1],
                                                            'b': training_graphs[client][t_idx][1][2],
                                                            'l': training_graphs[client][t_idx][1][3]},
                                                      ref_id=str(t_idx))
                pors.append(por)
        end_time = time.time()
        print((end_time-start_time) / (n_cycles + len(training_graphs[client])))
        domains[client]['pors'] = pors

        plot_pors(pors=pors, name=f'association_{client}')
        plot_pors(pors=pors, name=f'temporal_{client}')

        sam_dict = domains[client]['fabric'].sams[f'association_{client}']['sam'].to_dict(decode=True)

        rn_data = [t_data[2] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam=sam_dict,
                 raw_data=cycle_data,
                 xyz_types=['r_r', 'g_g', 'b_b'],
                 colour_nodes=None)

        start_idx = 50

        por_qry = domains[client]['fabric'].query_temporal(context_sdrs=[{'r': training_graphs[client][q_idx][1][0],
                                                                          'g': training_graphs[client][q_idx][1][1],
                                                                          'b': training_graphs[client][q_idx][1][2],
                                                                          'l': training_graphs[client][q_idx][1][3]}
                                                                         for q_idx in range(start_idx, start_idx + 4)])
        actual_sgm = por_qry[f'temporal_{client}']['sgm'].to_dict(decode=True)
        target_sgm_r = training_graphs[client][start_idx + 4][1][0].to_dict(decode=True)
        target_sgm_g = training_graphs[client][start_idx + 4][1][1].to_dict(decode=True)
        target_sgm_b = training_graphs[client][start_idx + 4][1][2].to_dict(decode=True)
        target_sgm_l = training_graphs[client][start_idx + 4][1][3].to_dict(decode=True)

        print(f'finished {client}')





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

        t_sdr = SGM()
        t_sdr.add_encoding('r', value=record['r'], encoder=numeric_encoder)
        t_sdr.add_encoding('g', value=record['g'], encoder=numeric_encoder)
        t_sdr.add_encoding('b', value=record['b'], encoder=numeric_encoder)
        t_sdr.add_encoding('label', value=record['label'], encoder=label_encoder)
        r_data = [record['r'], record['g'], record['b']]

        training_graphs[record['client']].append((record['trade_id'], t_sdr, r_data))

    n_cycles = 1

    domains = {}
    for client in training_graphs:
        pors = []
        domains[client] = {'fabric': SAMFabric(domain=client,
                                               assoc_similarity_threshold=0.75,
                                               assoc_anomaly_threshold_factor=3.0,
                                               temporal_similarity_threshold=0.75,
                                               temporal_anomaly_threshold_factor=3.0,
                                               similarity_ema_alpha=0.1,
                                               learn_rate_decay=0.3,
                                               prune_threshold=0.01,
                                               prune_neurons=False,
                                               )}

        domains[client]['fabric'].create_sam(name=client,
                                             similarity_threshold=0.75,
                                             anomaly_threshold_factor=3.0,
                                             search_types={'r', 'g', 'b'},
                                             learn_types={'r', 'g', 'b', 'label'},
                                             create_temporal=True)

        start_time = time.time()
        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = domains[client]['fabric'].train(sgms={client: training_graphs[client][t_idx][1]},
                                                      ref_id=str(t_idx))
                pors.append(por)
        end_time = time.time()
        print((end_time-start_time) / (n_cycles + len(training_graphs[client])))
        domains[client]['pors'] = pors

        plot_pors(pors=pors, name=client)
        plot_pors(pors=pors, name=f'temporal_{client}')

        sam_dict = domains[client]['fabric'].sams[client]['sam'].to_dict(decode=True)

        rn_data = [t_data[2] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)

        plot_sam(sam=sam_dict,
                 raw_data=cycle_data,
                 xyz_types=['r', 'g', 'b'],
                 colour_nodes=None)

        start_idx = 50
        por_qry = domains[client]['fabric'].query_temporal(context_sdrs=[{client: training_graphs[client][q_idx][1]} for q_idx in range(start_idx, start_idx + 4)])
        actual_sdr = por_qry[f'temporal_{client}']['sgm'].to_dict(decode=True)
        target_sdr = training_graphs[client][start_idx + 4][1].to_dict(decode=True)

        print(f'finished {client}')


if __name__ == '__main__':

    colours_association_test()
    colours_test()
