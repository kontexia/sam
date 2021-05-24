#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_distributed_representation import SDR
from src.value_encoder import ValueEncoder
from src.sam_fabric import SAM
from examples.sam_viz import plot_pors, plot_sam
import pandas as pd


def train():
    file_name = 'training_40.csv'

    df = pd.read_csv(file_name, header=0, delimiter=',')
    train_data = df.to_dict('records')

    print('raw data record:', train_data[:1])

    # create an encoder for each column's value
    #
    encoder = ValueEncoder(name='dq_encoder',
                           n_bits=40,
                           enc_size=2048,
                           numeric_step=1.0)

    regional_sam_params = {
        # an incoming training sdr must be at least 70% similar to a neuron to be mapped to it
        'similarity_threshold': 0.7,

        # neurons that are at least 63% (0.7 * 0.9) similar to the incoming sdr are considered to be in the same community
        'community_factor': 0.9,

        # the level below which a weight is considered zero and will be deleted
        'prune_threshold': 0.01,

        # a set of enc_type tuples to be used in learning - setting to None implies all enc_types will be learned
        'activation_enc_keys': None,
    }

    association_sam_params = {
        # an incoming training sdr must be at least 50% similar to a neuron to be mapped to it
        'similarity_threshold': 0.6,

        # neurons that are at least 45% (0.5 * 0.9) similar to the incoming sdr are considered to be in the same community
        'community_factor': 0.9,

        # the level below which a weight is considered zero and will be deleted
        'prune_threshold': 0.01,

        # a set of enc_type tuples to be used in learning - setting to None implies all enc_types will be learned
        'activation_enc_keys': None}

    # using default means any region will have these parameters
    #
    region_params = {'default': regional_sam_params}

    training_graphs = []
    training_raw_data = []
    sam_fabric = SAM(spatial_params=region_params, association_params=association_sam_params)
    for record in train_data:

        row_sdrs = {}
        raw_row = []
        for column in record:

            # ignore row_id column as its always unique
            if column not in ['Row_id', 'Order_id']:
                col_sgm = SDR()
                col_sgm.add_encoding(enc_key=(column,), value=record[column], encoder=encoder)
                row_sdrs[column] = col_sgm
                if column != 'Product':
                    raw_row.append(record[column])
        training_graphs.append(row_sdrs)
        training_raw_data.append(raw_row)
    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam_fabric.train(sdrs=training_graphs[t_idx])
        pors.append(por)

    sam_dict = sam_fabric.to_dict(decode=True)

    """
    plot_sam(sam_region=sam_dict['association'],
             raw_data=training_raw_data,
             xyz_types=[('RGB_Red',), ('RGB_Green',), ('RGB_Blue',)],
             colour_nodes=None,
             temporal_key=0)
    """
    plot_pors([por['_association'] for por in pors])

    file_name = 'test_40.csv'

    df = pd.read_csv(file_name, header=0, delimiter=',')
    test_data = df.to_dict('records')
    testing_graphs = []
    testing_raw_data = []
    for record in test_data:

        row_sdrs = {}
        raw_row = []
        for column in record:

            # ignore row_id column as its always unique
            if column not in ['Row_id', 'Order_id']:
                col_sgm = SDR()
                col_sgm.add_encoding(enc_key=(column,), value=record[column], encoder=encoder)
                row_sdrs[column] = col_sgm
                if column != 'Product':
                    raw_row.append(record[column])
        testing_graphs.append(row_sdrs)
        testing_raw_data.append(raw_row)

    for t_idx in range(len(testing_graphs)):
        por = sam_fabric.train(sdrs=testing_graphs[t_idx])
        pors.append(por)

    sam_dict = sam_fabric.to_dict(decode=True)

    """
    plot_sam(sam_region=sam_dict['association'],
             raw_data=training_raw_data,
             xyz_types=[('RGB_Red',), ('RGB_Green',), ('RGB_Blue',)],
             colour_nodes=None,
             temporal_key=0)
    """
    plot_pors([por['_association'] for por in pors])

    print('finished')


if __name__ == '__main__':
    train()
