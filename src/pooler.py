#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.neural_graph import NeuralGraph
from src.sparse_distributed_representation import SDR


class Pooler(object):

    def __init__(self,
                 name,
                 similarity_threshold: float = 0.7,
                 community_factor: float = 0.7,
                 prune_threshold=0.05):

        # Note: relate the decay factor of the learning rate to the similarity factor
        #
        self.neurons = NeuralGraph()

        self.name = name
        self.similarity_threshold = similarity_threshold
        self.community_threshold = community_factor * similarity_threshold
        self.prune_threshold = prune_threshold
        self.n_update = 0
        self.avg_bmu_similarity = 0.0
        self.variance_bmu_similarity = 0.0

    def to_dict(self, decode: bool = True):
        dict_pooler = {'name': self.name,
                       'similarity_threshold': self.similarity_threshold,
                       'prune_threshold': self.prune_threshold,
                       'n_update': self.n_update,
                       'avg_bmu_similarity': self.avg_bmu_similarity,
                       'variance_bmu_similarity': self.variance_bmu_similarity,
                       'neural_graph': self.neurons.to_dict(decode=decode)}

        return dict_pooler

    def get_neuron(self, neuron_key, decode: bool = False):

        neuron = self.neurons.get_neuron(neuron_key=neuron_key, decode=decode)

        return neuron

    def learn_pattern(self, sdr, activation_enc_keys: set = None):

        self.n_update += 1

        por = {'name': self.name,
               'n_update': self.n_update}

        train_sdr = SDR(sdr)

        # get the activated neurons
        #
        activated_neurons = self.neurons.feed_forward_pattern(sdr=train_sdr, activation_enc_keys=activation_enc_keys)

        bmu_similarity = 0.0
        if len(activated_neurons) == 0 or activated_neurons[0]['similarity'] < self.similarity_threshold:

            # add a new neuron
            #
            neuron_key = self.neurons.add_neuron(train_sdr, created=self.n_update)
            por['new_neuron_key'] = neuron_key
            if len(activated_neurons) > 0:
                bmu_similarity = activated_neurons[0]['similarity']

            # create an entry for the new neuron
            #
            activated_neurons.insert(0, {'neuron_key': neuron_key,
                                         'similarity': 1.0,
                                         'last_updated': self.n_update,
                                         'sdr': SDR()})
        else:

            # only the top neuron to learn the pattern
            #
            self.neurons.learn_pattern(activated_neuron_list=activated_neurons[:1],
                                       sdr=train_sdr,
                                       updated=self.n_update,
                                       prune_threshold=self.prune_threshold)

            bmu_similarity = activated_neurons[0]['similarity']

        # filter down to the community and update the community edges
        #
        community_neurons = [n for n in activated_neurons if n['similarity'] >= self.community_threshold]
        self.neurons.update_communities(community_neurons)

        self.avg_bmu_similarity += (bmu_similarity - self.avg_bmu_similarity) * 0.3
        self.variance_bmu_similarity = ((pow(bmu_similarity - self.avg_bmu_similarity, 2)) - self.variance_bmu_similarity) * 0.3

        por['bmu_similarity'] = bmu_similarity
        por['avg_bmu_similarity'] = self.avg_bmu_similarity

        if self.variance_bmu_similarity > 0.0001:
            por['std_bmu_similarity'] = pow(self.variance_bmu_similarity, 0.5)
        else:
            por['std_bmu_similarity'] = 0.0

        por['nos_neurons'] = len(self.neurons.neurons)
        por['activations'] = activated_neurons
        return por

    def query(self, sdr, similarity_threshold: float = None, decode=False) -> dict:
        if similarity_threshold is None:
            # return all neurons that have some similarity
            #
            similarity_threshold = 0.01

        # get the activated neurons
        #
        activated_neurons = self.neurons.feed_forward_pattern(sdr=sdr, decode=decode)

        por = {'activations': [activation
                               for activation in activated_neurons
                               if activation['similarity'] >= similarity_threshold]}

        if len(por['activations']) > 0:
            por['bmu_key'] = por['activations'][0]['neuron_key']
            por['sdr'] = self.neurons.neurons[por['bmu_key']]['pattern_sdr']
            if decode:
                por['sdr'] = por['sdr'].decode(self.neurons.neurons[por['bmu_key']]['n_bmu'])

        return por
