#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
import cython
from src.sparse_generalised_memory import SGM

@cython.cclass
class SparseNeurons(object):
    n_bits = cython.declare(int, visibility='public')
    next_neuron_key = cython.declare(int, visibility='public')
    learn_rate = cython.declare(float, visibility='public')
    bit_to_neuron = cython.declare(dict, visibility='public')
    neuron_to_bit = cython.declare(dict, visibility='public')

    def __init__(self,
                 n_bits:int = 40,
                 learn_rate: float = 0.6):

        self.n_bits = n_bits

        self.learn_rate = learn_rate

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

        neuron_key: int
        prop: str

        d_sparse_neurons = {'n_bits': self.n_bits,
                            'learn_rate': self.learn_rate,
                            'bit_to_neuron': deepcopy(self.bit_to_neuron),
                            'next_neuron_key': self.next_neuron_key,
                            'neuron_to_bit': {neuron_key: {prop: self.neuron_to_bit[neuron_key][prop] if prop != 'sgm' else self.neuron_to_bit[neuron_key][prop].to_dict(decode=decode)
                                                           for prop in self.neuron_to_bit[neuron_key]}
                                              for neuron_key in self.neuron_to_bit}}

        return d_sparse_neurons

    def feed_forward(self,
                     sgm: SGM,
                     activation_temporal_keys: set = None,
                     activation_enc_keys: set = None) -> list:

        activated_neurons: dict = {}
        activated_neuron_list: list

        temporal_key: int
        enc_key: str
        normalisation_factor: int
        bit: int

        if activation_temporal_keys is None:
            activation_temporal_keys = set(self.bit_to_neuron.keys()) | set(sgm.encoding.keys())

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

        for temporal_key in sgm.encoding:

            if temporal_key not in self.bit_to_neuron:
                self.bit_to_neuron[temporal_key] = {}

            # activate only via the activation temporal keys
            #
            if activation_temporal_keys is None or temporal_key in activation_temporal_keys:

                for enc_key in sgm.encoding[temporal_key]:

                    if enc_key not in self.bit_to_neuron[temporal_key]:
                        self.bit_to_neuron[temporal_key][enc_key] = {}

                    # activate only via the activation enc_keys
                    #
                    if activation_enc_keys is None or enc_key in activation_enc_keys:

                        for bit in sgm.encoding[temporal_key][enc_key]:

                            if bit not in self.bit_to_neuron[temporal_key][enc_key]:
                                self.bit_to_neuron[temporal_key][enc_key][bit] = set()

                            for neuron_key in self.bit_to_neuron[temporal_key][enc_key][bit]:

                                # activation depends on weight of connection between bit and neuron and magnitude of incoming Bit
                                #
                                if neuron_key not in activated_neurons:

                                    activated_neurons[neuron_key] = (sgm.encoding[temporal_key][enc_key][bit] *
                                                                     self.neuron_to_bit[neuron_key]['sgm'].encoding[temporal_key][enc_key][bit])
                                else:
                                    activated_neurons[neuron_key] += (sgm.encoding[temporal_key][enc_key][bit] *
                                                                      self.neuron_to_bit[neuron_key]['sgm'].encoding[temporal_key][enc_key][bit])

        # normalise activation and place in list so we can sort
        #
        activated_neuron_list = [{'neuron_key': neuron_key,
                                  'activation': activated_neurons[neuron_key],
                                  'similarity': activated_neurons[neuron_key] / normalisation_factor if normalisation_factor > 0 else 1.0}
                                 for neuron_key in activated_neurons]

        activated_neuron_list.sort(key=lambda x: x['similarity'], reverse=True)

        return activated_neuron_list

    def update_bit_to_neuron(self, neuron_key: int):
        temporal_key: int
        enc_key: str
        bit: int

        neuron_sgm: SGM = self.neuron_to_bit[neuron_key]['sgm']

        for temporal_key in neuron_sgm.encoding:
            if temporal_key not in self.bit_to_neuron:
                self.bit_to_neuron[temporal_key] = {}

            for enc_key in neuron_sgm.encoding[temporal_key]:
                if enc_key not in self.bit_to_neuron[temporal_key]:
                    self.bit_to_neuron[temporal_key][enc_key] = {}

                for bit in neuron_sgm.encoding[temporal_key][enc_key]:
                    if bit not in self.bit_to_neuron[temporal_key][enc_key]:
                        self.bit_to_neuron[temporal_key][enc_key][bit] = {neuron_key}
                    else:
                        self.bit_to_neuron[temporal_key][enc_key][bit].add(neuron_key)

    def add_neuron(self, sgm: SGM) -> int:
        
        # get next neuron key to use
        #
        neuron_key: int = self.next_neuron_key
        self.next_neuron_key += 1
        
        # add the neuron to bit mapping
        #
        self.neuron_to_bit[neuron_key] = {'sgm': SGM(sgm),
                                          'n_bmu': 1,
                                          'sum_similarity': 1.0,
                                          'avg_similarity': 1.0,
                                          'n_runner_up': 0}

        self.update_bit_to_neuron(neuron_key)

        return neuron_key

    def learn(self, activated_neuron_list: list, sgm: SGM, learn_enc_keys: set = None):

        idx: int
        neuron_key: int
        learn_rate: float

        for idx in range(len(activated_neuron_list)):
            neuron_key = activated_neuron_list[idx]['neuron_key']

            if idx == 0:

                # keep track of the number of times this neurons has been the bmu
                #
                self.neuron_to_bit[neuron_key]['n_bmu'] += 1

                # the learning rate for the bmu
                #
                learn_rate = self.learn_rate

                # keep track of the similarity of data mapped to this generalised memory
                #
                self.neuron_to_bit[neuron_key]['sum_similarity'] += activated_neuron_list[idx]['similarity']
                self.neuron_to_bit[neuron_key]['avg_similarity'] = self.neuron_to_bit[neuron_key]['sum_similarity'] / self.neuron_to_bit[neuron_key]['n_bmu']

            else:
                # not a bmu so learning rate must be less than bmu
                #
                learn_rate = self.learn_rate * 0.1

                # keep track of the number of times this neurons has been the updated when not a bmu
                #
                self.neuron_to_bit[neuron_key]['n_runner_up'] += 1

            # update the neurons's sparse generalised memory with the incoming data
            # note always learn whole of temporal memory but not always all enc_keys
            #
            self.neuron_to_bit[neuron_key]['sgm'].learn(sgm=sgm, learn_rate=learn_rate, learn_enc_keys=learn_enc_keys)

            # finally make sure the bits are connected to this neuron
            #
            self.update_bit_to_neuron(neuron_key)


if __name__ == '__main__':

    t_1 = SGM()
    t_1.encoding = {0: {'a': {0: 1.0, 1: 1.0}}}
    t_2 = SGM()
    t_2.encoding = {0: {'a': {2: 1.0, 1: 1.0}}}

    fabric = SparseNeurons(n_bits=2)

    fabric.add_neuron(sgm=t_1)
    fabric.add_neuron(sgm=t_2)

    t_3 = SGM()
    t_3.encoding = {0: {'a': {2: 1.0, 3: 1.0}}}

    activated_neurons = fabric.feed_forward(sgm=t_3)

    fabric.learn(activated_neuron_list=activated_neurons, sgm=t_3)

    print('finished')