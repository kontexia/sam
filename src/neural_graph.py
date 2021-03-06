#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
import cython
from src.sparse_distributed_representation import SDR, TEMPORAL_IDX, ENC_IDX



@cython.cclass
class NeuralGraph(object):
    next_neuron_key = cython.declare(cython.int, visibility='public')
    pattern_to_neuron = cython.declare(dict, visibility='public')
    neurons = cython.declare(dict, visibility='public')

    def __init__(self):

        # models the edges from sdr bits to neurons
        #
        self.pattern_to_neuron = {}

        # the map of neurons which generalise sdrs
        #
        self.neurons = {}

        # the next valid neuron id
        #
        self.next_neuron_key = 0

    def to_dict(self, decode: bool = True) -> dict:

        neuron_key: cython.int
        prop: str

        dict_neural_graph = {'pattern_to_neuron': deepcopy(self.pattern_to_neuron),
                             'next_neuron_key': self.next_neuron_key,
                             'neurons': {neuron_key: {prop: self.neurons[neuron_key][prop] if 'sdr' not in prop else self.neurons[neuron_key][prop].to_dict(decode=decode, max_bit_weight=self.neurons[neuron_key]['n_bmu'])
                                                      for prop in self.neurons[neuron_key]}
                                         for neuron_key in self.neurons}}

        return dict_neural_graph

    def feed_forward_pattern(self,
                             sdr: SDR,
                             activation_temporal_keys: set = None,
                             activation_enc_keys: set = None,
                             decode: bool = False) -> list:

        activated_neurons: dict = {}
        activated_neuron_list: list

        normalisation_factor: float
        sdr_key: tuple
        bit: cython.int
        neuron_key: cython.int

        for sdr_key in sdr.encoding:
            if ((activation_temporal_keys is None or sdr_key[TEMPORAL_IDX] in activation_temporal_keys) and
                    (activation_enc_keys is None or sdr_key[ENC_IDX] in activation_enc_keys)):

                # need to normalise by the total weight of all bits associated with the sdr_key
                #
                normalisation_factor = sum([sdr.encoding[sdr_key][bit] for bit in sdr.encoding[sdr_key]])

                for bit in sdr.encoding[sdr_key]:

                    # if this key and bit have been mapped then process
                    #
                    if sdr_key in self.pattern_to_neuron and bit in self.pattern_to_neuron[sdr_key]:
                        for neuron_key in self.pattern_to_neuron[sdr_key][bit]:

                            # activation depends on weight of connection between bit and neuron and magnitude of incoming Bit
                            #
                            if neuron_key not in activated_neurons:

                                activated_neurons[neuron_key] = (sdr.encoding[sdr_key][bit] * self.neurons[neuron_key]['pattern_sdr'].encoding[sdr_key][bit] /
                                                                 (self.neurons[neuron_key]['n_bmu'] * normalisation_factor))
                            else:
                                activated_neurons[neuron_key] += (sdr.encoding[sdr_key][bit] * self.neurons[neuron_key]['pattern_sdr'].encoding[sdr_key][bit] /
                                                                  (self.neurons[neuron_key]['n_bmu'] * normalisation_factor))

        # now normalise across the number of sdr_keys to give a similarity between 0.0 and 1.0
        #
        normalisation_factor = float(len(sdr.encoding))

        # normalise activation and place in list so we can sort
        #
        activated_neuron_list = [{'neuron_key': neuron_key,
                                  'similarity': max(min(activated_neurons[neuron_key] / normalisation_factor, 1.0), 0.0),
                                  'last_updated': self.neurons[neuron_key]['last_updated'],
                                  'sdr': self.neurons[neuron_key]['pattern_sdr'] if not decode else self.neurons[neuron_key]['pattern_sdr'].decode(self.neurons[neuron_key]['n_bmu'])}
                                 for neuron_key in activated_neurons]

        # sort by most similar with tie break bias towards most recently updated
        #
        activated_neuron_list.sort(key=lambda x: (x['similarity'], x['last_updated']), reverse=True)

        return activated_neuron_list

    def update_pattern_to_neuron(self, neuron_key: cython.int):
        bit: cython.int
        sdr_key: tuple

        neuron_sdr: SDR = self.neurons[neuron_key]['pattern_sdr']

        for sdr_key in neuron_sdr.encoding:
            if sdr_key not in self.pattern_to_neuron:
                self.pattern_to_neuron[sdr_key] = {}

            # remove the connection if bit not in neuron_sdr - it may have been pruned
            #
            for bit in self.pattern_to_neuron[sdr_key]:
                if bit not in neuron_sdr.encoding[sdr_key]:
                    self.pattern_to_neuron[sdr_key][bit].discard(neuron_key)

            # make sure all bits in neuron_sdr have a connection to this neuron
            #
            for bit in neuron_sdr.encoding[sdr_key]:
                if bit not in self.pattern_to_neuron[sdr_key]:
                    self.pattern_to_neuron[sdr_key][bit] = {neuron_key}
                else:
                    self.pattern_to_neuron[sdr_key][bit].add(neuron_key)

    def add_neuron(self, sdr: SDR, created: cython.int) -> cython.int:

        # get next neuron key to use
        #
        neuron_key: cython.int = self.next_neuron_key
        self.next_neuron_key += 1

        # add the neuron to bit mapping
        #
        self.neurons[neuron_key] = {'pattern_sdr': SDR(sdr),
                                    'n_bmu': 1,
                                    'sum_similarity': 1.0,
                                    'avg_similarity': 1.0,
                                    'created': created,
                                    'last_updated': created,
                                    'community': set()}

        self.update_pattern_to_neuron(neuron_key)

        return neuron_key

    def learn_pattern(self, activated_neuron_list: list, sdr: SDR, updated: cython.int, learn_enc_keys: set = None, prune_threshold: float = 0.05):

        idx: cython.int
        neuron_key: cython.int
        learn_rate: float

        for idx in range(len(activated_neuron_list)):
            neuron_key = activated_neuron_list[idx]['neuron_key']

            # keep track of the number of times this neurons has been the bmu
            #
            self.neurons[neuron_key]['n_bmu'] += 1

            # keep track of when it was last updated
            #
            self.neurons[neuron_key]['last_updated'] = updated

            # keep track of the similarity of data mapped to this generalised memory
            #
            self.neurons[neuron_key]['sum_similarity'] += activated_neuron_list[idx]['similarity']
            self.neurons[neuron_key]['avg_similarity'] = self.neurons[neuron_key]['sum_similarity'] / self.neurons[neuron_key]['n_bmu']

            # update the neuron's sparse generalised memory with the incoming data
            # note always learn whole of temporal memory but not always all enc_keys
            #
            self.neurons[neuron_key]['pattern_sdr'].learn_frequency(sdr=sdr, learn_enc_keys=learn_enc_keys, min_frequency_prune=int(prune_threshold * self.neurons[neuron_key]['n_bmu']))

            # finally make sure the pattern bits are connected to this neuron
            #
            self.update_pattern_to_neuron(neuron_key)

    def update_communities(self, activated_neurons: list):

        idx_1: cython.int
        idx_2: cython.int

        for idx_1 in range(len(activated_neurons)):
            for idx_2 in range(len(activated_neurons)):
                if idx_1 != idx_2:
                    self.neurons[activated_neurons[idx_1]['neuron_key']]['community'].add(activated_neurons[idx_2]['neuron_key'])

    def get_neuron(self, neuron_key: cython.int, decode: bool = False) -> dict:
        neuron = None
        if neuron_key in self.neurons:
            neuron = {prop: self.neurons[neuron_key][prop] if 'sdr' not in prop else self.neurons[neuron_key][prop].to_dict(decode=decode, max_bit_weight=self.neurons[neuron_key]['n_bmu'])
                      for prop in self.neurons[neuron_key]}
        return neuron


if __name__ == '__main__':

    t_1 = SDR()
    t_1.encoding = {(0, ('a',)): {0: 1.0, 1: 1.0}}
    t_2 = SDR()
    t_2.encoding = {(0, ('a',)): {2: 1.0, 1: 1.0}}

    n_graph = NeuralGraph()

    n_graph.add_neuron(sdr=t_1, created=0)
    n_graph.add_neuron(sdr=t_2, created=0)

    t_3 = SDR()
    t_3.encoding = {(0, ('a',)): {2: 1.0, 3: 1.0}}

    activated_neurons = n_graph.feed_forward_pattern(sdr=t_3)

    n_graph.learn(activated_neuron_list=activated_neurons, sdr=t_3)

    print('finished')