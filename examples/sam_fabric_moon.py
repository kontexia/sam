#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_distributed_representation import SDR
from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder
from src.sam_fabric import SAMFabric
from examples.sam_viz import plot_pors, plot_sam


from sklearn.datasets import make_moons


def moon_test():

    # get moon raw data with 1% noise and 200 samples
    #
    data_set, labels = make_moons(n_samples=200,
                                  noise=0.01,
                                  random_state=0)

    # create a numeric encoder and label encoder
    #
    numeric_encoder = NumericEncoder(min_step=0.005,
                                     n_bits=40,
                                     enc_size=2048,
                                     seed=123)

    label_encoder = CategoryEncoder(n_bits=40,
                                    enc_size=2048,
                                    seed=456)

    # construct a training set of SDRs encoding the x and y values of the 2D moons and
    # encoding the label
    #
    training_graphs = []

    for idx in range(len(data_set)):

        # the xy_sdr represents the sparse encoding of a 2D point
        #
        xy_sdr = SDR()

        # here enc_key identifies an edge to an x value
        #
        xy_sdr.add_encoding(enc_key=('x',), value=data_set[idx][0], encoder=numeric_encoder)

        # here enc_key identifies an edge to a y value
        #
        xy_sdr.add_encoding(enc_key=('y',), value=data_set[idx][1], encoder=numeric_encoder)

        # the label_sdr represents the sparse encoding of the label
        #
        label_sdr = SDR()
        label_sdr.add_encoding(enc_key=('label',), value=str(labels[idx]), encoder=label_encoder)

        training_graphs.append({'xy': xy_sdr, 'label': label_sdr})

    # the xy_sdr and label_sdr will be trained in different regions of the SAMFabric
    # but associated with each other because they occur at the same time
    # The sam_params is a default config for each region of the SAMFabric
    #
    sam_params = {
        # an incoming training sdr must be at least 70% similar to a neuron to be mapped to it
        'similarity_threshold': 0.7,

        # neurons that are at least 63% (0.7 * 0.9) similar to the incoming sdr are considered to be in the same community
        'community_factor': 0.9,

        # setting a temporal_learning_rate to 1.0 effectively turns off temporal learning as 100% is learned from the
        # incoming sdr and 0% (1 - 1.0) is remembered from the previous SDRs
        'temporal_learning_rate': 1.0,

        # the level below which a weight is considered zero and will be deleted
        'prune_threshold': 0.01,

        # a set of enc_type tuples to be used in learning - settin to None implies all enc_types will be learned
        'activation_enc_keys': None,

        # the learning rate of associative connections between neurons of different regions
        'association_learn_rate': 0.6}

    sam_fabric = SAMFabric(association_params=sam_params)

    # we have the possibility of having different configurations per region
    #
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


if __name__ == '__main__':

    moon_test()
