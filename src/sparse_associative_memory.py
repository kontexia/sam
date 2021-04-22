#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sparse_neurons import SparseNeurons
from src.sparse_generalised_memory import SGM


class SAM(object):

    def __init__(self,
                 name,
                 similarity_threshold: float = 0.75,
                 learn_rate: float = 0.6,
                 learn_temporal: bool = False,
                 n_bits: int = 40,
                 prune_threshold=0.01):

        self.neurons = SparseNeurons(n_bits=n_bits, learn_rate=learn_rate)
        self.name = name
        self.similarity_threshold = similarity_threshold
        self.learn_rate = learn_rate
        self.learn_temporal = learn_temporal
        self.temporal_sgm = SGM()
        self.prune_threshold = prune_threshold

        self.n_update = 0
        self.avg_bmu_similarity = 0.0
        self.variance_bmu_similarity = 0.0

    def to_dict(self, decode: bool = True):
        d_sam = {'name': self.name,
                 'similarity_threshold': self.similarity_threshold,
                 'learn_rate': self.learn_rate,
                 'learn_temporal': self.learn_temporal,
                 'prune_threshold': self.prune_threshold,
                 'temporal_sgm': self.temporal_sgm.to_dict(decode=decode),
                 'n_update': self.n_update,
                 'avg_bmu_similarity': self.avg_bmu_similarity,
                 'variance_bmu_similarity': self.variance_bmu_similarity,
                 'neurons': self.neurons.to_dict(decode=decode)}
        return d_sam

    def train(self, sgm, activation_enc_keys: set = None):

        self.n_update += 1

        por = {'name': self.name,
               'n_update': self.n_update}

        train_sgm = SGM(sgm)

        # if we are learning temporal sequences then add in the temporal memory
        #
        if self.learn_temporal:
            train_sgm.copy_from(sgm=self.temporal_sgm, from_temporal_key=0, to_temporal_key=1)

        activated_neurons = self.neurons.feed_forward(sgm=train_sgm,
                                                      activation_enc_keys=activation_enc_keys)

        bmu_similarity = 0.0
        if len(activated_neurons) == 0 or activated_neurons[0]['similarity'] < self.similarity_threshold:
            neuron_key = self.neurons.add_neuron(train_sgm)
            por['new_neuron_key'] = neuron_key
            if len(activated_neurons) > 0:
                bmu_similarity = activated_neurons[0]['similarity']
                activated_neurons = activated_neurons[:1]
        else:
            activated_neurons = [activation for activation in activated_neurons if activation['similarity'] >= self.similarity_threshold]
            self.neurons.learn(activated_neuron_list=activated_neurons, sgm=train_sgm, prune_threshold=self.prune_threshold)
            bmu_similarity = activated_neurons[0]['similarity']

        self.avg_bmu_similarity += (bmu_similarity - self.avg_bmu_similarity) * 0.3
        self.variance_bmu_similarity = ((pow(bmu_similarity - self.avg_bmu_similarity, 2)) - self.variance_bmu_similarity) * 0.3

        if self.learn_temporal:
            self.temporal_sgm.learn(sgm=sgm, learn_rate=self.learn_rate, learn_enc_keys=activation_enc_keys, prune_threshold=self.prune_threshold)

        por['bmu_similarity']= bmu_similarity
        por['avg_bmu_similarity'] = self.avg_bmu_similarity
        if self.variance_bmu_similarity > 0.0001:
            por['std_bmu_similarity'] = pow(self.variance_bmu_similarity, 0.5)
        else:
            por['std_bmu_similarity'] = 0.0

        por['nos_neurons'] = len(self.neurons.neuron_to_bit)
        por['activations'] = activated_neurons
        return por

    def query(self, sgm, similarity_threshold: float = None, decode=False) -> dict:
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        query_sgm = SGM()

        if isinstance(sgm, list) and self.learn_temporal:
            lstm = None
            for idx in range(len(sgm)):
                if idx == 0:
                    lstm = SGM(sgm[0])
                else:
                    lstm.learn(sgm=sgm[idx], learn_rate=self.learn_rate)

            query_sgm.copy_from(sgm=lstm, from_temporal_key=0, to_temporal_key=1)
        else:
            query_sgm = sgm
        activated_neurons = self.neurons.feed_forward(sgm=query_sgm)

        por = {'activations': [activation
                               for activation in activated_neurons
                               if activation['similarity'] >= similarity_threshold]}
        if len(por['activations']) > 0:
            por['bmu_key'] = por['activations'][0]['neuron_key']
            por['sgm'] = self.neurons.neuron_to_bit[por['bmu_key']]['sgm']
            if decode:
                por['sgm'] = por['sgm'].decode()

        return por

