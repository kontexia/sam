#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
import cython
from src.sparse_distributed_representation import SDR
from math import exp


@cython.cclass
class Neurons(object):
    n_bits = cython.declare(cython.int, visibility='public')
    next_neuron_key = cython.declare(cython.int, visibility='public')
    learning_rate_decay_factor = cython.declare(cython.int, visibility='public')
    bit_to_neuron = cython.declare(dict, visibility='public')
    neuron_to_bit = cython.declare(dict, visibility='public')

    def __init__(self,
                 n_bits: cython.int = 40,
                 learning_rate_decay_factor: cython.int = 10):

        self.n_bits = n_bits

        self.learning_rate_decay_factor = pow(learning_rate_decay_factor, 2)

        # this maps bits to neurons
        #
        self.bit_to_neuron = {}
        
        # this maps neurons to bits
        #
        self.neuron_to_bit = {}
        
        # the next valid neuron
        #
        self.next_neuron_key = 0

    def to_dict(self, decode: bool = True) -> dict:

        neuron_key: cython.int
        prop: str

        d_neurons = {'n_bits': self.n_bits,
                     'learning_rate_decay_factor': self.learning_rate_decay_factor,
                     'bit_to_neuron': deepcopy(self.bit_to_neuron),
                     'next_neuron_key': self.next_neuron_key,
                     'neuron_to_bit': {neuron_key: {prop: self.neuron_to_bit[neuron_key][prop] if prop != 'sdr' else self.neuron_to_bit[neuron_key][prop].to_dict(decode=decode)
                                                    for prop in self.neuron_to_bit[neuron_key]}
                                       for neuron_key in self.neuron_to_bit}}

        return d_neurons

    def feed_forward(self,
                     sdr: SDR,
                     activation_temporal_keys: set = None,
                     activation_enc_keys: set = None) -> list:

        activated_neurons: dict = {}
        activated_neuron_list: list

        temporal_key: cython.int
        enc_key: str
        normalisation_factor: cython.int
        bit: cython.int

        if activation_temporal_keys is None:
            activation_temporal_keys = set(self.bit_to_neuron.keys()) | set(sdr.encoding.keys())

        if activation_enc_keys is None:
            activation_enc_keys = ({enc_key
                                    for temporal_key in self.bit_to_neuron
                                    if temporal_key in activation_temporal_keys
                                    for enc_key in self.bit_to_neuron[temporal_key]} |
                                   {enc_key
                                    for temporal_key in self.bit_to_neuron
                                    if temporal_key in activation_temporal_keys
                                    for enc_key in self.bit_to_neuron[temporal_key]})

        normalisation_factor = len(activation_temporal_keys) * len(activation_enc_keys) * self.n_bits

        for temporal_key in sdr.encoding:

            if temporal_key not in self.bit_to_neuron:
                self.bit_to_neuron[temporal_key] = {}

            # activate only via the activation temporal keys
            #
            if activation_temporal_keys is None or temporal_key in activation_temporal_keys:

                for enc_key in sdr.encoding[temporal_key]:

                    if enc_key not in self.bit_to_neuron[temporal_key]:
                        self.bit_to_neuron[temporal_key][enc_key] = {}

                    # activate only via the activation enc_keys
                    #
                    if activation_enc_keys is None or enc_key in activation_enc_keys:

                        for bit in sdr.encoding[temporal_key][enc_key]:

                            if bit not in self.bit_to_neuron[temporal_key][enc_key]:
                                self.bit_to_neuron[temporal_key][enc_key][bit] = set()

                            for neuron_key in self.bit_to_neuron[temporal_key][enc_key][bit]:

                                # activation depends on weight of connection between bit and neuron and magnitude of incoming Bit
                                #
                                if neuron_key not in activated_neurons:

                                    activated_neurons[neuron_key] = (sdr.encoding[temporal_key][enc_key][bit] *
                                                                     self.neuron_to_bit[neuron_key]['sdr'].encoding[temporal_key][enc_key][bit])
                                else:
                                    activated_neurons[neuron_key] += (sdr.encoding[temporal_key][enc_key][bit] *
                                                                      self.neuron_to_bit[neuron_key]['sdr'].encoding[temporal_key][enc_key][bit])

        # normalise activation and place in list so we can sort
        #
        activated_neuron_list = [{'neuron_key': neuron_key,
                                  'activation': activated_neurons[neuron_key],
                                  'similarity': activated_neurons[neuron_key] / normalisation_factor if normalisation_factor > 0 else 1.0}
                                 for neuron_key in activated_neurons]

        activated_neuron_list.sort(key=lambda x: x['similarity'], reverse=True)

        return activated_neuron_list

    def update_bit_to_neuron(self, neuron_key: cython.int):
        temporal_key: cython.int
        enc_key: str
        bit: cython.int

        neuron_sdr: SDR = self.neuron_to_bit[neuron_key]['sdr']

        for temporal_key in neuron_sdr.encoding:
            if temporal_key not in self.bit_to_neuron:
                self.bit_to_neuron[temporal_key] = {}

            for enc_key in neuron_sdr.encoding[temporal_key]:
                if enc_key not in self.bit_to_neuron[temporal_key]:
                    self.bit_to_neuron[temporal_key][enc_key] = {}
                else:
                    # remove the connection if bit not in neuron_sdr - it may have been pruned
                    #
                    for bit in self.bit_to_neuron[temporal_key][enc_key]:
                        if bit not in neuron_sdr.encoding[temporal_key][enc_key]:
                            self.bit_to_neuron[temporal_key][enc_key][bit].discard(neuron_key)

                # make sure all bits in neuron_sdr have a connection to this neuron
                #
                for bit in neuron_sdr.encoding[temporal_key][enc_key]:
                    if bit not in self.bit_to_neuron[temporal_key][enc_key]:
                        self.bit_to_neuron[temporal_key][enc_key][bit] = {neuron_key}
                    else:
                        self.bit_to_neuron[temporal_key][enc_key][bit].add(neuron_key)

    def add_neuron(self, sdr: SDR) -> cython.int:
        
        # get next neuron key to use
        #
        neuron_key: cython.int = self.next_neuron_key
        self.next_neuron_key += 1
        
        # add the neuron to bit mapping
        #
        self.neuron_to_bit[neuron_key] = {'sdr': SDR(sdr),
                                          'n_bmu': 1,
                                          'learn_rate': 1.0,
                                          'sum_similarity': 1.0,
                                          'avg_similarity': 1.0}

        self.update_bit_to_neuron(neuron_key)

        return neuron_key

    def learn(self, activated_neuron_list: list, sdr: SDR, learn_enc_keys: set = None, prune_threshold: float = 0.01):

        idx: cython.int
        neuron_key: cython.int
        learn_rate: float

        for idx in range(len(activated_neuron_list)):
            neuron_key = activated_neuron_list[idx]['neuron_key']

            # keep track of the number of times this neurons has been the bmu
            #
            self.neuron_to_bit[neuron_key]['n_bmu'] += 1

            # the learning rate for the activated neuron depends inversely on the number of times this neuron has been mapped to
            #
            self.neuron_to_bit[neuron_key]['learn_rate'] = exp(-self.neuron_to_bit[neuron_key]['n_bmu'] * self.learning_rate_decay_factor)

            # keep track of the similarity of data mapped to this generalised memory
            #
            self.neuron_to_bit[neuron_key]['sum_similarity'] += activated_neuron_list[idx]['similarity']
            self.neuron_to_bit[neuron_key]['avg_similarity'] = self.neuron_to_bit[neuron_key]['sum_similarity'] / self.neuron_to_bit[neuron_key]['n_bmu']

            # update the neurons's sparse generalised memory with the incoming data
            # note always learn whole of temporal memory but not always all enc_keys
            #
            self.neuron_to_bit[neuron_key]['sdr'].learn(sdr=sdr, learn_rate=self.neuron_to_bit[neuron_key]['learn_rate'], learn_enc_keys=learn_enc_keys, prune_threshold=prune_threshold)

            # finally make sure the bits are connected to this neuron
            #
            self.update_bit_to_neuron(neuron_key)


if __name__ == '__main__':

    t_1 = SDR()
    t_1.encoding = {0: {'a': {0: 1.0, 1: 1.0}}}
    t_2 = SDR()
    t_2.encoding = {0: {'a': {2: 1.0, 1: 1.0}}}

    fabric = Neurons(n_bits=2)

    fabric.add_neuron(sdr=t_1)
    fabric.add_neuron(sdr=t_2)

    t_3 = SDR()
    t_3.encoding = {0: {'a': {2: 1.0, 3: 1.0}}}

    activated_neurons = fabric.feed_forward(sdr=t_3)

    fabric.learn(activated_neuron_list=activated_neurons, sdr=t_3)

    print('finished')