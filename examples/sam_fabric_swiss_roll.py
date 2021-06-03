#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_distributed_representation import SDR
from src.numeric_encoder import NumericEncoder
from src.sam_fabric import SAM
from examples.sam_viz import plot_sam


from sklearn.datasets import make_swiss_roll


def swiss_roll_test():

    # get swiss roll raw data with 1% noise and 200 samples
    #
    data_set, labels = make_swiss_roll(n_samples=200,
                                       noise=0.01,
                                       random_state=0)

    # create a numeric encoder
    #
    numeric_encoder = NumericEncoder(min_step=0.2,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    # construct a training set of SDRs encoding the x y and z values of the 3D swiss roll
    #
    training_graphs = []

    for idx in range(len(data_set)):

        xyz_sdr = SDR()

        # here enc_key identifies an edge to an x value
        #
        xyz_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)

        # here enc_key identifies an edge to an y value
        #
        xyz_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)

        # here enc_key identifies an edge to an z value
        #
        xyz_sdr.add_encoding(enc_key=('z',), value=data_set[idx][2], encoder=numeric_encoder)

        training_graphs.append({'xyz': xyz_sdr})

    # configuration for the sam fabric
    #
    sam_params = {'xyz':
        {
            # an incoming training sdr must be at least 85% similar to a neuron to be mapped to it
            'similarity_threshold': 0.8,

            # neurons that are at least 76.5% (0.85 * 0.9) similar to the incoming sdr are considered to be in the same community
            'community_factor': 0.95,

            # the level below which a weight is considered zero and will be deleted
            'prune_threshold': 0.01,

            # a set of enc_type tuples to be used in learning - settin to None implies all enc_types will be learned
            'activation_enc_keys': None
        }
    }

    sam_fabric = SAM(spatial_params=sam_params)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = sam_fabric.train(sdrs=training_graphs[t_idx])
        pors.append(por)

    sam_dict = sam_fabric.to_dict(decode=True)

    plot_sam(sam_region=sam_dict['xyz'],
             raw_data=data_set,
             xyz_types=[('x',), ('y',), ('z',)],
             colour_nodes=None,
             title='Swiss Roll')


if __name__ == '__main__':

    swiss_roll_test()
