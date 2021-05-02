#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
from src.sparse_associative_memory import SAM
from src.sparse_distributed_representation import SDR
from src.value_encoder import ValueEncoder
from src.sparse_am_viz import plot_pors, plot_sam


def train():
    file_name = 'training_40.csv'

    df = pd.read_csv(file_name, header=0, delimiter=',')
    train_data = df.to_dict('records')

    print('raw data record:', train_data[:1])

    encoder = ValueEncoder(name='dq_encoder',
                           n_bits=40,
                           enc_size=2048,
                           numeric_step=1.0)

    training_graphs = {'column': {}, 'row': []}

    sams = {'row': {'sam': SAM(name='row',
                               similarity_threshold=0.8,
                               temporal_learn_rate=1.0,
                               n_bits=80),
                    'por': []
                    },
            'column': {}
            }
    for record in train_data:

        t_sgm = SDR()
        t_sgm.add_encoding(enc_key='RGB_Red', value=record['RGB_Red'], encoder=encoder)
        t_sgm.add_encoding(enc_key='RGB_Green', value=record['RGB_Green'], encoder=encoder)
        t_sgm.add_encoding(enc_key='RGB_Blue', value=record['RGB_Blue'], encoder=encoder)
        t_sgm.add_encoding(enc_key='Product', value=record['Product'], encoder=encoder)
        r_data = [record['RGB_Red'], record['RGB_Green'], record['RGB_Blue'], record['Product']]

        training_graphs['row'].append((record['Row_id'], t_sgm, r_data))

        for column in record:
            if column != 'Row_id':
                if column not in sams['column']:
                    sams['column'][column] = {'sam': SAM(name=column,
                                                         similarity_threshold=0.8,
                                                         temporal_learn_rate=1.0,
                                                         n_bits=80),
                                              'por': []}

                t_sgm = SDR()
                t_sgm.add_encoding(enc_key=column, value=record[column], encoder=encoder)
                if column not in training_graphs['column']:
                    training_graphs['column'][column] = []
                training_graphs['column'][column].append((record['Row_id'], t_sgm, record[column]))

    row_data = []
    for record in training_graphs['row']:
        por = sams['row']['sam'].train(sdr=record[1])
        sams['row']['por'].append(por)
        row_data.append(record[2][:3])
    row_dict = sams['row']['sam'].to_dict(decode=True)
    plot_sam(sam=row_dict,
             raw_data=row_data,
             xyz_types=['RGB_Red', 'RGB_Green', 'RGB_Blue'],
             colour_nodes=None,
             temporal_key=0)

    plot_pors(sams['row']['por'])

    column_dict = {}
    for column in training_graphs['column']:
        for record in training_graphs['column'][column]:
            por = sams['column'][column]['sam'].train(sdr=record[1])
            sams['column'][column]['por'].append(por)
        column_dict[column] = sams['column'][column]['sam'].to_dict(decode=True)
    print('finished')

    file_name = 'test_40.csv'

    df = pd.read_csv(file_name, header=0, delimiter=',')
    test_data = df.to_dict('records')
    test_graphs = {'column': {}, 'row': []}

    for record in test_data:

        t_sgm = SDR()
        t_sgm.add_encoding(enc_key='RGB_Red', value=record['RGB_Red'], encoder=encoder)
        t_sgm.add_encoding(enc_key='RGB_Green', value=record['RGB_Green'], encoder=encoder)
        t_sgm.add_encoding(enc_key='RGB_Blue', value=record['RGB_Blue'], encoder=encoder)
        t_sgm.add_encoding(enc_key='Product', value=record['Product'], encoder=encoder)
        r_data = [record['RGB_Red'], record['RGB_Green'], record['RGB_Blue'], record['Product']]

        test_graphs['row'].append((record['Row_id'], t_sgm, r_data))

        for column in record:
            if column != 'Row_id':
                t_sgm = SDR()
                t_sgm.add_encoding(enc_key=column, value=record[column], encoder=encoder)
                if column not in test_graphs['column']:
                    test_graphs['column'][column] = []
                test_graphs['column'][column].append((record['Row_id'], t_sgm, record[column]))

    row_data = []
    for record in test_graphs['row']:
        por = sams['row']['sam'].train(sdr=record[1])
        sams['row']['por'].append(por)
        row_data.append(record[2][:3])
    row_dict = sams['row']['sam'].to_dict(decode=True)
    plot_sam(sam=row_dict,
             raw_data=row_data,
             xyz_types=['RGB_Red', 'RGB_Green', 'RGB_Blue'],
             colour_nodes=None,
             temporal_key=0)

    plot_pors(sams['row']['por'])

    column_dict = {}
    for column in test_graphs['column']:
        for record in test_graphs['column'][column]:
            por = sams['column'][column]['sam'].train(sdr=record[1])
            sams['column'][column]['por'].append(por)
        column_dict[column] = sams['column'][column]['sam'].to_dict(decode=True)
    print('finished')


if __name__ == '__main__':
    train()