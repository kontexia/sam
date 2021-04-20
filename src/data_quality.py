#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
from src.sparse_associative_memory import SAM
from src.sparse_generalised_memory import SGM
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder
from src.sparse_am_viz import plot_pors, plot_sam



def train():
    file_name = 'training.csv'

    raw_data = []
    df = pd.read_csv(file_name, header=0, delimiter=',')
    raw_data = df.to_dict('records')

    print('raw data record:', raw_data[:1])

    numeric_encoder = NumericEncoder(min_step=2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = StringEncoder(n_bits=40,
                                  enc_size=2048,
                                  seed=456)

    training_graphs = {'column': {}, 'row': []}

    sams = {'row': {'sam': SAM(name='row',
                               similarity_threshold=0.7,
                               learn_rate=0.6,
                               learn_temporal=False,
                               n_bits=40),
                    'por': []
                    },
            'column': {}
            }
    for record in raw_data:

        t_sgm = SGM()
        t_sgm.add_encoding(enc_key='RGB_Red', value=record['RGB_Red'], encoder=numeric_encoder)
        t_sgm.add_encoding(enc_key='RGB_Green', value=record['RGB_Green'], encoder=numeric_encoder)
        t_sgm.add_encoding(enc_key='RGB_Blue', value=record['RGB_Blue'], encoder=numeric_encoder)
        t_sgm.add_encoding(enc_key='Product', value=record['Product'], encoder=label_encoder)
        r_data = [record['RGB_Red'], record['RGB_Green'], record['RGB_Blue'], record['Product']]

        training_graphs['row'].append((record['Row_id'], t_sgm, r_data))

        for column in ['Client', 'Product']:
            if column not in sams['column']:
                sams['column'][column] = {'sam': SAM(name=column,
                                                     similarity_threshold=0.7,
                                                     learn_rate=0.6,
                                                     learn_temporal=False,
                                                     n_bits=40),
                                          'por': []}

            t_sgm = SGM()
            t_sgm.add_encoding(enc_key=column, value=record[column], encoder=label_encoder)
            if column not in training_graphs['column']:
                training_graphs['column'][column] = []
            training_graphs['column'][column].append((record['Row_id'], t_sgm, record[column]))

        for column in ['RGB_Red', 'RGB_Green', 'RGB_Blue']:
            if column not in sams['column']:
                sams['column'][column] = {'sam': SAM(name=column,
                                                     similarity_threshold=0.7,
                                                     learn_rate=0.6,
                                                     learn_temporal=False,
                                                     n_bits=40),
                                          'por': []}

            t_sgm = SGM()
            t_sgm.add_encoding(enc_key=column, value=record[column], encoder=numeric_encoder)
            if column not in training_graphs['column']:
                training_graphs['column'][column] = []
            training_graphs['column'][column].append((record['Row_id'], t_sgm, record[column]))

    for record in training_graphs['row']:
        por = sams['row']['sam'].train(sgm=record[1])
        sams['row']['por'].append(por)

    row_dict = sams['row']['sam'].to_dict(decode=True)
    column_dict = {}
    for column in training_graphs['column']:
        for record in training_graphs['column'][column]:
            por = sams['column'][column]['sam'].train(sgm=record[1])
            sams['column'][column]['por'].append(por)
        column_dict[column] = sams['column'][column]['sam'].to_dict(decode=True)
    print('finished')


if __name__ == '__main__':
    train()