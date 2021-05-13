#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.neurons import NeuralGraph
from src.sparse_distributed_representation import SDR

"""
class SAM_v1(object):

    def __init__(self,
                 name,
                 similarity_threshold: float = 0.7,
                 community_threshold: float = 0.7,
                 temporal_learn_rate: float = 1.0,
                 prune_threshold=0.01):

        # Note: relate the decay factor of the learning rate to the similarity factor
        #
        self.neurons = Neurons(learning_rate_decay_factor=(1 - similarity_threshold))
        self.name = name
        self.similarity_threshold = similarity_threshold
        self.community_threshold = community_threshold * similarity_threshold
        self.temporal_learn_rate = temporal_learn_rate
        self.temporal_sdr = SDR()
        self.prune_threshold = prune_threshold

        self.n_update = 0
        self.avg_bmu_similarity = 0.0
        self.variance_bmu_similarity = 0.0

    def to_dict(self, decode: bool = True):
        d_sam = {'name': self.name,
                 'similarity_threshold': self.similarity_threshold,
                 'temporal_learn_rate': self.temporal_learn_rate,
                 'prune_threshold': self.prune_threshold,
                 'temporal_sdr': self.temporal_sdr.to_dict(decode=decode),
                 'n_update': self.n_update,
                 'avg_bmu_similarity': self.avg_bmu_similarity,
                 'variance_bmu_similarity': self.variance_bmu_similarity,
                 'neurons': self.neurons.to_dict(decode=decode)}
        return d_sam

    def train(self, sdr, activation_enc_keys: set = None):

        self.n_update += 1

        por = {'name': self.name,
               'n_update': self.n_update}

        train_sdr = SDR(sdr)

        # if we are learning temporal sequences then add in the temporal memory
        #
        if self.temporal_learn_rate < 1.0:
            train_sdr.copy_from(sdr=self.temporal_sdr, from_temporal_key=0, to_temporal_key=1)

        activated_neurons = self.neurons.feed_forward(sdr=train_sdr,
                                                      activation_enc_keys=activation_enc_keys)

        bmu_similarity = 0.0
        if len(activated_neurons) == 0 or activated_neurons[0]['similarity'] < self.similarity_threshold:
            neuron_key = self.neurons.add_neuron(train_sdr)
            por['new_neuron_key'] = neuron_key
            if len(activated_neurons) > 0:
                bmu_similarity = activated_neurons[0]['similarity']

            activated_neurons.insert(0, {'neuron_key': neuron_key, 'similarity': 1.0})
        else:
            self.neurons.learn(activated_neuron_list=activated_neurons[:1], sdr=train_sdr, prune_threshold=self.prune_threshold)
            bmu_similarity = activated_neurons[0]['similarity']

        # filter down to the community and update the community edges
        #
        activated_neurons = [n for n in activated_neurons if n['similarity'] >= self.community_threshold]
        self.neurons.update_communities(activated_neurons)

        self.avg_bmu_similarity += (bmu_similarity - self.avg_bmu_similarity) * 0.3
        self.variance_bmu_similarity = ((pow(bmu_similarity - self.avg_bmu_similarity, 2)) - self.variance_bmu_similarity) * 0.3

        if self.temporal_learn_rate < 1.0:
            self.temporal_sdr.learn(sdr=sdr, learn_rate=self.temporal_learn_rate, learn_enc_keys=activation_enc_keys, prune_threshold=self.prune_threshold)

        por['bmu_similarity'] = bmu_similarity
        por['avg_bmu_similarity'] = self.avg_bmu_similarity
        if self.variance_bmu_similarity > 0.0001:
            por['std_bmu_similarity'] = pow(self.variance_bmu_similarity, 0.5)
        else:
            por['std_bmu_similarity'] = 0.0

        por['nos_neurons'] = len(self.neurons.neuron_to_bit)
        por['activations'] = activated_neurons
        return por

    def query(self, sdr, similarity_threshold: float = None, decode=False) -> dict:
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        query_sdr = SDR()

        if isinstance(sdr, list) and self.temporal_learn_rate < 1.0:
            temporal_sdr = None
            for idx in range(len(sdr)):
                if idx == 0:
                    temporal_sdr = SDR(sdr[0])
                else:
                    temporal_sdr.learn(sdr=sdr[idx], learn_rate=self.temporal_learn_rate)

            query_sdr.copy_from(sdr=temporal_sdr, from_temporal_key=0, to_temporal_key=1)
        else:
            query_sdr = sdr
        activated_neurons = self.neurons.feed_forward(sdr=query_sdr)

        por = {'activations': [activation
                               for activation in activated_neurons
                               if activation['similarity'] >= similarity_threshold]}
        if len(por['activations']) > 0:
            por['bmu_key'] = por['activations'][0]['neuron_key']
            por['sdr'] = self.neurons.neuron_to_bit[por['bmu_key']]['sdr']
            if decode:
                por['sdr'] = por['sdr'].decode()

        return por
"""


class SAMRegion(object):

    def __init__(self,
                 name,
                 similarity_threshold: float = 0.7,
                 community_threshold: float = 0.7,
                 temporal_learn_rate: float = 1.0,
                 prune_threshold=0.01):

        # Note: relate the decay factor of the learning rate to the similarity factor
        #
        self.neural_graph = NeuralGraph(learning_rate_decay_factor=(1 - similarity_threshold))
        self.name = name
        self.similarity_threshold = similarity_threshold
        self.community_threshold = community_threshold * similarity_threshold
        self.temporal_learn_rate = temporal_learn_rate
        self.temporal_sdr = SDR()
        self.prune_threshold = prune_threshold

        self.n_update = 0
        self.avg_bmu_similarity = 0.0
        self.variance_bmu_similarity = 0.0

    def to_dict(self, decode: bool = True):
        dict_sam = {'name': self.name,
                    'similarity_threshold': self.similarity_threshold,
                    'temporal_learn_rate': self.temporal_learn_rate,
                    'prune_threshold': self.prune_threshold,
                    'temporal_sdr': self.temporal_sdr.to_dict(decode=decode),
                    'n_update': self.n_update,
                    'avg_bmu_similarity': self.avg_bmu_similarity,
                    'variance_bmu_similarity': self.variance_bmu_similarity,
                    'neural_graph': self.neural_graph.to_dict(decode=decode)}
        return dict_sam

    def get_neuron(self, neuron_key):
        return self.neural_graph.get_neuron(neuron_key=neuron_key)

    def learn_pattern(self, sdr, activation_enc_keys: set = None):

        self.n_update += 1

        por = {'name': self.name,
               'n_update': self.n_update}

        train_sdr = SDR(sdr)

        # if we are learning temporal sequences then add in the temporal memory
        #
        if self.temporal_learn_rate < 1.0:
            train_sdr.copy_from(sdr=self.temporal_sdr, from_temporal_key=0, to_temporal_key=1)

        # get the activated neurons
        #
        activated_neurons = self.neural_graph.feed_forward_pattern(sdr=train_sdr,
                                                                   activation_enc_keys=activation_enc_keys)

        bmu_similarity = 0.0
        if len(activated_neurons) == 0 or activated_neurons[0]['similarity'] < self.similarity_threshold:

            # add a new neuron
            #
            neuron_key = self.neural_graph.add_neuron(train_sdr, created=self.n_update)
            por['new_neuron_key'] = neuron_key
            if len(activated_neurons) > 0:
                bmu_similarity = activated_neurons[0]['similarity']

            # create an entry for the nes neuron
            #
            activated_neurons.insert(0, {'neuron_key': neuron_key,
                                         'similarity': 1.0,
                                         'last_updated': self.n_update,
                                         'association_sdr': SDR()})
        else:

            # top neuron to learn the pattern
            #
            self.neural_graph.learn_pattern(activated_neuron_list=activated_neurons[:1],
                                            sdr=train_sdr,
                                            updated=self.n_update,
                                            prune_threshold=self.prune_threshold)

            bmu_similarity = activated_neurons[0]['similarity']

        # filter down to the community and update the community edges
        #
        activated_neurons = [n for n in activated_neurons if n['similarity'] >= self.community_threshold]
        self.neural_graph.update_communities(activated_neurons)

        self.avg_bmu_similarity += (bmu_similarity - self.avg_bmu_similarity) * 0.3
        self.variance_bmu_similarity = ((pow(bmu_similarity - self.avg_bmu_similarity, 2)) - self.variance_bmu_similarity) * 0.3

        # construct the temporal pattern activations
        #
        if self.temporal_learn_rate < 1.0:
            self.temporal_sdr.learn(sdr=sdr,
                                    learn_rate=self.temporal_learn_rate,
                                    learn_enc_keys=activation_enc_keys,
                                    prune_threshold=self.prune_threshold)

        por['bmu_similarity'] = bmu_similarity
        por['avg_bmu_similarity'] = self.avg_bmu_similarity
        if self.variance_bmu_similarity > 0.0001:
            por['std_bmu_similarity'] = pow(self.variance_bmu_similarity, 0.5)
        else:
            por['std_bmu_similarity'] = 0.0

        por['nos_neurons'] = len(self.neural_graph.neurons)
        por['activations'] = activated_neurons
        return por

    def query(self, sdr, similarity_threshold: float = None, decode=False) -> dict:
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        query_sdr = SDR()

        # if we are being asked to query a temporal sequence then construct an appropriate SDR
        #
        if isinstance(sdr, list) and self.temporal_learn_rate < 1.0:
            temporal_sdr = None
            for idx in range(len(sdr)):
                if idx == 0:
                    temporal_sdr = SDR(sdr[0])
                else:
                    temporal_sdr.learn(sdr=sdr[idx], learn_rate=self.temporal_learn_rate)

            query_sdr.copy_from(sdr=temporal_sdr, from_temporal_key=0, to_temporal_key=1)
        else:
            query_sdr = sdr

        # get the activated neurons
        #
        activated_neurons = self.neural_graph.feed_forward_pattern(sdr=query_sdr)

        por = {'activations': [activation
                               for activation in activated_neurons
                               if activation['similarity'] >= similarity_threshold]}

        if len(por['activations']) > 0:
            por['bmu_key'] = por['activations'][0]['neuron_key']
            por['sdr'] = self.neural_graph.neuron_to_bit[por['bmu_key']]['sdr']
            if decode:
                por['sdr'] = por['sdr'].decode()

        return por

    def associate(self, neuron_key, sdr, learn_rate: float = 0.5):

        self.neural_graph.learn_association(neuron_key=neuron_key,
                                            sdr=sdr,
                                            learn_rate=learn_rate,
                                            prune_threshold=self.prune_threshold)

