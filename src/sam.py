#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sgm import SGM
from typing import Optional, Tuple
from copy import deepcopy


class SAM(object):
    def __init__(self,
                 name,
                 similarity_threshold: float = 0.75,
                 community_threshold: float = 0.70,
                 anomaly_threshold_factor: float = 3.0,
                 similarity_ema_alpha: float = 0.3,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01):
        self.name = name
        self.anomaly_threshold_factor: float = anomaly_threshold_factor
        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.anomaly_threshold: float = 0.0
        self.anomalies: dict = {}
        self.prune_threshold: float = prune_threshold
        self.neurons = {}
        self.update_id: int = 0
        self.next_neuron_id: int = 0
        self.last_bmu_key: Optional[str] = None
        self.ema_similarity: Optional[float] = None
        self.ema_variance: float = 0.0
        self.motif_threshold: float = 1.0
        self.motifs: dict = {}
        self.updated: bool = True
        self.similarity_threshold = similarity_threshold
        self.community_threshold = community_threshold
        self.communities = {}
        self.community_max_count = 0

    def add_neuron(self, sgm: SGM) -> str:
        neuron_key = f'{self.next_neuron_id}'
        self.next_neuron_id += 1

        update_id = str(self.update_id)

        self.neurons[neuron_key] = {'sam': self.name,
                                    'uid': neuron_key,
                                    'sgm': sgm,
                                    'created': update_id,
                                    'ema_similarity': None,
                                    'n_bmu': 1,
                                    'last_bmu': update_id,
                                    'n_runner_up': 0,
                                    'last_runner_up': None,
                                    'activation': 1.0,
                                    'learn_rate': 1.0,
                                    'nn': {}}

        self.last_bmu_key = neuron_key

        return neuron_key

    def to_dict(self, decode: bool = False) -> dict:
        d_sam = {'name': self.name,
                 'similarity_ema_alpha': self.similarity_ema_alpha,
                 'similarity_threshold': self.similarity_threshold,
                 'learn_rate_decay': self.learn_rate_decay,
                 'prune_threshold': self.prune_threshold,
                 'update_id': self.update_id,
                 'next_neuron_id': self.next_neuron_id,
                 'last_bmu_key': self.last_bmu_key,
                 'ema_similarity': self.ema_similarity,
                 'ema_variance': self.ema_variance,
                 'anomaly_threshold_factor': self.anomaly_threshold_factor,
                 'anomaly_threshold': self.anomaly_threshold,
                 'anomalies': self.anomalies,
                 'motif_threshold': self.motif_threshold,
                 'motifs': self.motifs,
                 'neurons': deepcopy(self.neurons),
                 }

        # replace neuron sdrs with dict (decoded as required)
        #
        for neuron_key in d_sam['neurons']:
            d_sam['neurons'][neuron_key]['sgm'] = d_sam['neurons'][neuron_key]['sgm'].to_dict(decode=decode)
            d_sam['neurons'][neuron_key]['nn'] = {nn_key: d_sam['neurons'][neuron_key]['nn'][nn_key] / self.community_max_count
                                                  for nn_key in d_sam['neurons'][neuron_key]['nn']}

        return d_sam

    def check_anomaly_motif(self, bmu_key: str, bmu_similarity: float, ref_id: str, new_neuron: bool = False) -> Tuple[bool, bool]:

        # update the ema error and variance using the slow_alpha
        #
        if self.ema_similarity is None:
            self.ema_similarity = bmu_similarity
        else:
            self.ema_similarity += (bmu_similarity - self.ema_similarity) * self.similarity_ema_alpha

        self.ema_variance += (pow((bmu_similarity - self.ema_similarity), 2) - self.ema_variance) * self.similarity_ema_alpha

        # record breaches of anomaly threshold
        #
        report: dict = {'bmu_key': bmu_key, 'mapped': self.update_id, 'similarity': bmu_similarity, 'ref_id': ref_id}
        anomaly = False
        motif = False

        # determine if anomaly or motif detected
        #
        if bmu_similarity < self.anomaly_threshold and new_neuron:
            self.anomalies[str(self.update_id)] = report
            anomaly = True
        elif self.motif_threshold is not None and bmu_similarity >= self.motif_threshold:
            self.motifs[str(self.update_id)] = report
            motif = True

        # update threshold for next training data
        #
        stdev = pow(self.ema_variance, 0.5)
        self.anomaly_threshold = max(self.ema_similarity - (self.anomaly_threshold_factor * stdev), 0.0)
        self.motif_threshold = min(self.ema_similarity + (2.0 * stdev), 1.0)

        return anomaly, motif

    def train(self, sgm, ref_id: str, search_types: set, learn_types: set) -> dict:
        por = {'sam': self.name,
               'ref_id': ref_id,
               'bmu_key': None,
               'bmu_similarity': 0.0,
               'new_neuron_key': None,
               'nn_neurons': [],
               'anomaly': False,
               'motif': False,
               'ema_similarity': self.ema_similarity,
               'ema_variance': self.ema_variance,
               'anomaly_threshold': self.anomaly_threshold,
               'motif_threshold': self.motif_threshold,
               'activations': {},
               'deleted_neuron_key': None}

        self.update_id += 1
        self.updated = True

        if len(self.neurons) == 0:
            # add new neuron
            #
            new_neuron_key = self.add_neuron(sgm=sgm)

            por['new_neuron_key'] = new_neuron_key
            por['activations'][new_neuron_key] = self.neurons[new_neuron_key]['activation']
        else:

            # calc the similarity of the sgm to the existing neurons
            #
            similarities = [(neuron_key,
                             self.neurons[neuron_key]['sgm'].similarity(sgm=sgm, search_types=search_types),
                             self.neurons[neuron_key]['n_bmu'])
                            for neuron_key in self.neurons]

            # sort in descending order of similarity and number of times bmu
            #
            similarities.sort(key=lambda x: (x[1]['similarity'], x[2]), reverse=True)

            # the bmu is the closest and thus the top of the list
            #
            bmu_key = similarities[0][0]
            bmu_similarity = similarities[0][1]['similarity']

            por['bmu_key'] = bmu_key
            por['bmu_similarity'] = bmu_similarity

            por['activations'][bmu_key] = bmu_similarity

            # get neurons within the community threshold
            #
            community = [similarities[idx][0] for idx in range(len(similarities)) if similarities[idx][1]['similarity'] >= self.community_threshold]

            # if the similarity is smaller than the threshold then add a new neuron
            #
            if bmu_similarity < self.similarity_threshold:

                # add new neuron
                #
                new_neuron_key = self.add_neuron(sgm=sgm)

                por['new_neuron_key'] = new_neuron_key
                por['activations'][new_neuron_key] = self.neurons[new_neuron_key]['activation']

                community.append(new_neuron_key)

            else:

                # the data is close enough to the bmu to be mapped
                # so update the bmu neuron attributes
                #
                self.neurons[bmu_key]['n_bmu'] += 1
                self.neurons[bmu_key]['last_bmu'] = self.update_id

                # a neuron's similarity for mapped data is the exponential moving average of the similarity.
                #
                if self.neurons[bmu_key]['ema_similarity'] is None:
                    self.neurons[bmu_key]['ema_similarity'] = bmu_similarity
                else:
                    self.neurons[bmu_key]['ema_similarity'] += ((bmu_similarity - self.neurons[bmu_key]['ema_similarity']) * self.similarity_ema_alpha)

                # decay the learning rate so that this neuron learns more slowly the more it gets mapped too
                #
                self.neurons[bmu_key]['learn_rate'] -= self.neurons[bmu_key]['learn_rate'] * self.learn_rate_decay

                # learn the generalised graph
                #
                self.neurons[bmu_key]['sgm'].learn(sgm=sgm,
                                                   learn_rate=self.neurons[bmu_key]['learn_rate'],
                                                   learn_types=learn_types)

                updated_neurons = set()

                updated_neurons.add(bmu_key)

                if len(similarities) > 1:

                    # process the rest of the neurons
                    #
                    for nn_idx in range(1, len(similarities)):

                        nn_key = similarities[nn_idx][0]
                        nn_similarity = similarities[nn_idx][1]['similarity']
                        self.neurons[nn_key]['activation'] = nn_similarity
                        por['activations'][nn_key] = nn_similarity

                        # if the neurons is similar enough then learn
                        #
                        if nn_similarity >= self.similarity_threshold:

                            updated_neurons.add(nn_key)
                            por['nn_neurons'].append({'nn_key': nn_key,
                                                      'nn_similarity': nn_similarity})

                            self.neurons[nn_key]['n_runner_up'] += 1
                            self.neurons[nn_key]['last_runner_up'] = self.update_id

                            # the learning rate for a neighbour needs to be much less that the bmu - hence the product of learning rates and 0.1 factor
                            #
                            nn_learn_rate = self.neurons[bmu_key]['learn_rate'] * self.neurons[nn_key]['learn_rate'] * 0.1

                            # learn the sgm
                            #
                            self.neurons[nn_key]['sgm'].learn(sgm=sgm,
                                                              learn_rate=nn_learn_rate,
                                                              learn_types=learn_types)

            for neuron_key_1 in community:
                for neuron_key_2 in community:
                    if neuron_key_1 != neuron_key_2:
                        if neuron_key_2 not in self.neurons[neuron_key_1]['nn']:
                            self.neurons[neuron_key_1]['nn'][neuron_key_2] = 1
                        else:
                            self.neurons[neuron_key_1]['nn'][neuron_key_2] += 1
                            if self.neurons[neuron_key_1]['nn'][neuron_key_2] > self.community_max_count:
                                self.community_max_count = self.neurons[neuron_key_1]['nn'][neuron_key_2]

            anomaly, motif = self.check_anomaly_motif(bmu_key=bmu_key, bmu_similarity=bmu_similarity, ref_id=ref_id, new_neuron=por['new_neuron_key'])
            por['anomaly'] = anomaly
            por['motif'] = motif

        por['nos_neurons'] = len(self.neurons)

        return por

    def query(self, sgm, bmu_only: bool = True) -> dict:

        # get the types to search for
        #
        search_types = sgm.get_enc_types()

        # calc the distance of the sgm to the existing neurons
        #
        similarities = [(neuron_key,
                         self.neurons[neuron_key]['sgm'].similarity(sgm=sgm, search_types=search_types),
                         self.neurons[neuron_key]['n_bmu'],
                         self.neurons[neuron_key]['sgm'])
                        for neuron_key in self.neurons]

        # sort in descending order of similarity and number of times bmu
        #
        similarities.sort(key=lambda x: (x[1]['similarity'], x[2]), reverse=True)

        # get closest neuron and all other 'activated neurons'
        #
        activated_neurons = [similarities[n_idx]
                             for n_idx in range(len(similarities))
                             if n_idx == 0 or similarities[n_idx][1]['similarity'] >= self.community_threshold]

        por = {'sam': self.name}

        if len(activated_neurons) > 0:

            # select the bmu
            #
            if bmu_only or len(activated_neurons) == 1:
                por['sgm'] = activated_neurons[0][3]

                por['neurons'] = [{'neuron_key': n[0], 'similarity': n[1]['similarity']} for n in activated_neurons]

            # else create a weighted average of neurons
            #
            else:
                por['sgm'] = SGM()
                por['neurons'] = []
                for n in activated_neurons:
                    por['sgm'].merge(sgm=n[3], weight=n[1]['similarity'])
                    por['neurons'].append({'neuron_key': n[0], 'similarity': n[1]['similarity']})

        return por


if __name__ == '__main__':

    from src.string_encoder import StringEncoder
    from src.numeric_encoder import NumericEncoder

    ng = SAM(name='test',
             similarity_threshold=0.75,
             anomaly_threshold_factor=6.0,
             similarity_ema_alpha=0.1,
             learn_rate_decay=0.3,
             prune_threshold=0.01,
             prune_neurons=False)

    str_enc = StringEncoder(n_bits=40,
                            enc_size=2048,
                            seed=12345)

    numeric_enc = NumericEncoder(min_step=1.0,
                                 n_bits=40,
                                 enc_size=2048,
                                 seed=12345)

    t_1 = SGM()
    t_1.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_1.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_1.add_encoding(enc_type='Volume', value=100, encoder=numeric_enc)

    p1 = ng.train(sgm=t_1, ref_id='t_1', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_2 = SGM()
    t_2.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_2.add_encoding(enc_type='Platform', value='B', encoder=str_enc)
    t_2.add_encoding(enc_type='Volume', value=50, encoder=numeric_enc)

    p2 = ng.train(sgm=t_2, ref_id='t_2', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_3 = SGM()
    t_3.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_3.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_3.add_encoding(enc_type='Volume', value=75, encoder=numeric_enc)

    p3 = ng.train(sgm=t_3, ref_id='t_3', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_4 = SGM()
    t_4.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_4.add_encoding(enc_type='Platform', value='B', encoder=str_enc)
    t_4.add_encoding(enc_type='Volume', value=60, encoder=numeric_enc)

    p4 = ng.train(sgm=t_4, ref_id='t_4', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_5 = SGM()
    t_5.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_5.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_5.add_encoding(enc_type='Volume', value=110, encoder=numeric_enc)

    p5 = ng.train(sgm=t_5, ref_id='t_5', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    ng_dict = ng.to_dict(decode=True)

    t_6 = SGM()
    t_6.add_encoding(enc_type='Volume', value=90, encoder=numeric_enc)

    q_por_bmu = ng.query(sgm=t_6, bmu_only=True)
    q_por_bmu_decode = q_por_bmu['sgm'].decode()
    q_por_weav = ng.query(sgm=t_6, bmu_only=False)
    q_por_weav_decode = q_por_weav['sgm'].decode()

    print('finished')

