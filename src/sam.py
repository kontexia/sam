#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sdr import SDR
from typing import Optional, Set, Tuple
from copy import deepcopy


class SAM(object):
    def __init__(self,
                 name,
                 similarity_threshold: float = 0.75,
                 anomaly_threshold_factor: float = 4.0,
                 error_decay: float = 0.1,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01,
                 prune_neurons: bool = False):
        self.name = name
        self.anomaly_threshold_factor: float = anomaly_threshold_factor
        self.error_decay = error_decay
        self.learn_rate_decay = learn_rate_decay
        self.anomaly_threshold: float = 0.0
        self.anomalies: dict = {}
        self.prune_threshold: float = prune_threshold
        self.neurons = {}
        self.update_id: int = 0
        self.next_neuron_id: int = 0
        self.last_bmu_key: Optional[str] = None
        self.ema_error: Optional[float] = None
        self.ema_variance: float = 0.0
        self.motif_threshold: float = 0.0
        self.motifs: dict = {}
        self.prune_neurons: bool = prune_neurons
        self.updated: bool = True
        self.similarity_threshold = similarity_threshold

    def add_neuron(self, sdr: SDR) -> str:
        neuron_key = f'{self.next_neuron_id}'
        self.next_neuron_id += 1

        update_id = str(self.update_id)

        self.neurons[neuron_key] = {'sam': self.name,
                                    'uid': neuron_key,
                                    'sdr': sdr,
                                    'created': update_id,
                                    'ema_error': None,
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
                 'error_decay': self.error_decay,
                 'similarity_threshold': self.similarity_threshold,
                 'learn_rate_decay': self.learn_rate_decay,
                 'prune_threshold': self.prune_threshold,
                 'update_id': self.update_id,
                 'next_neuron_id': self.next_neuron_id,
                 'last_bmu_key': self.last_bmu_key,
                 'ema_error': self.ema_error,
                 'ema_variance': self.ema_variance,
                 'anomaly_threshold_factor': self.anomaly_threshold_factor,
                 'anomaly_threshold': self.anomaly_threshold,
                 'anomalies': self.anomalies,
                 'motif_threshold': self.motif_threshold,
                 'motifs': self.motifs,
                 'prune_neurons': self.prune_neurons,
                 'neurons': deepcopy(self.neurons)
                 }
        # replace neuron sdrs with dict (decoded as required)
        #
        for neuron_key in d_sam['neurons']:
            d_sam['neurons'][neuron_key]['sdr'] = d_sam['neurons'][neuron_key]['sdr'].to_dict(decode=decode)

        return d_sam

    def update_gas_error(self, bmu_key: str, bmu_distance: float, ref_id: str, new_neuron: bool = False) -> Tuple[bool, bool]:

        # update the ema error and variance using the slow_alpha
        #
        if self.ema_error is None:
            self.ema_error = bmu_distance
        else:
            self.ema_error += (bmu_distance - self.ema_error) * self.error_decay

        self.ema_variance += (pow((bmu_distance - self.ema_error), 2) - self.ema_variance) * self.error_decay

        # record breaches of anomaly threshold
        #
        report: dict = {'bmu_key': bmu_key, 'mapped': self.update_id, 'error': bmu_distance, 'ref_id': ref_id}
        anomaly = False
        motif = False

        # determine if anomaly or motif detected
        #
        if bmu_distance > self.anomaly_threshold and new_neuron:
            self.anomalies[str(self.update_id)] = report
            anomaly = True
        elif self.motif_threshold is not None and bmu_distance <= self.motif_threshold:
            self.motifs[str(self.update_id)] = report
            motif = True

        # update threshold for next training data
        #
        stdev = pow(self.ema_variance, 0.5)
        self.anomaly_threshold = self.ema_error + (self.anomaly_threshold_factor * stdev)
        self.motif_threshold = max(self.ema_error - (2.0 * stdev), 0.0)

        return anomaly, motif

    def train(self, sdr, ref_id: str, search_types: set, learn_types: set) -> dict:
        por = {'sam': self.name,
               'ref_id': ref_id,
               'bmu_key': None,
               'bmu_distance': None,
               'bmu_distance_threshold': 0.0,
               'bmu_similarity': 0.0,
               'new_neuron_key': None,
               'nn_neurons': [],
               'anomaly': False,
               'motif': False,
               'ema_error': self.ema_error,
               'ema_variance': self.ema_variance,
               'anomaly_threshold': self.anomaly_threshold,
               'motif_threshold': self.motif_threshold,
               'deleted_neuron_key': None}

        self.update_id += 1
        self.updated = True

        if len(self.neurons) == 0:
            # add new neuron
            #
            new_neuron_key = self.add_neuron(sdr=sdr)

            por['new_neuron_key'] = new_neuron_key
        else:

            # calc the distance of the sdr to the existing neurons
            #
            distances = [(neuron_key,
                          self.neurons[neuron_key]['sdr'].distance(sdr=sdr, search_types=search_types),
                          self.neurons[neuron_key]['n_bmu'])
                         for neuron_key in self.neurons]

            # sort in ascending order of actual distance and descending order of number of times bmu
            #
            distances.sort(key=lambda x: (x[1]['distance'], -x[2]))

            # the bmu is the closest and thus the top of the list
            #
            bmu_key = distances[0][0]
            bmu_distance = distances[0][1]['distance']

            por['bmu_key'] = bmu_key
            por['bmu_distance'] = bmu_distance
            por['bmu_similarity'] = distances[0][1]['similarity']
            por['bmu_distance_threshold'] = distances[0][1]['max_distance'] * (1 - self.similarity_threshold)

            # if the distance is larger than the neuron's threshold then add a new neuron
            #
            if bmu_distance > por['bmu_distance_threshold']:

                # add new neuron
                #
                new_neuron_key = self.add_neuron(sdr=sdr)

                por['new_neuron_key'] = new_neuron_key

                # connect the new neuron to the bmu neuron and remember the distance
                #
                self.neurons[bmu_key]['nn'][new_neuron_key] = bmu_distance
                self.neurons[new_neuron_key]['nn'][bmu_key] = bmu_distance

                if self.prune_neurons:
                    # get first neuron that has aged enough to be deleted
                    #
                    neuron_to_deactivate = []
                    for neuron_key in self.neurons:
                        if neuron_key not in [new_neuron_key, bmu_key]:

                            # decay the activation with rate the depends on its current learn_rate and the learn_rate_decay
                            #
                            self.neurons[neuron_key]['activation'] -= (self.neurons[neuron_key]['activation'] * self.learn_rate_decay * self.neurons[neuron_key]['learn_rate'])
                            if self.neurons[neuron_key]['activation'] < self.prune_threshold:
                                neuron_to_deactivate.append(neuron_key)

                                # only need first 1 so beak out of loop
                                #
                                break

                    if len(neuron_to_deactivate) > 0:
                        for nn_key in self.neurons[neuron_to_deactivate[0]]:
                            del self.neurons[nn_key]['nn'][neuron_to_deactivate[0]]
                        del self.neurons[neuron_to_deactivate[0]]
                        por['deleted_neuron_key'] = neuron_to_deactivate[0]

            else:

                # the data is close enough to the bmu to be mapped
                # so update the bmu neuron attributes
                #
                self.neurons[bmu_key]['n_bmu'] += 1
                self.neurons[bmu_key]['last_bmu'] = self.update_id

                # a neuron's error for mapped data is the exponential moving average of the distance.
                #
                if self.neurons[bmu_key]['ema_error'] is None:
                    self.neurons[bmu_key]['ema_error'] = bmu_distance
                else:
                    self.neurons[bmu_key]['ema_error'] += ((bmu_distance - self.neurons[bmu_key]['ema_error']) * self.error_decay)

                # decay the learning rate so that this neuron learns more slowly the more it gets mapped too
                #
                self.neurons[bmu_key]['learn_rate'] -= self.neurons[bmu_key]['learn_rate'] * self.learn_rate_decay

                # learn the generalised graph
                #
                self.neurons[bmu_key]['sdr'].learn(sdr=sdr,
                                                   learn_rate=self.neurons[bmu_key]['learn_rate'],
                                                   learn_types=learn_types)

                # reset the bmu activation to full strength
                #
                self.neurons[bmu_key]['activation'] = 1.0

                updated_neurons = set()

                updated_neurons.add(bmu_key)

                if len(distances) > 1:

                    nn_idx = 1
                    finished = False
                    while not finished:

                        nn_key = distances[nn_idx][0]
                        nn_distance = distances[nn_idx][1]['distance']

                        # if the neuron is close enough to the incoming data
                        #
                        nn_distance_threshold = distances[nn_idx][1]['max_distance'] * (1 - self.similarity_threshold)
                        if nn_distance <= nn_distance_threshold:

                            updated_neurons.add(nn_key)
                            por['nn_neurons'].append({'nn_distance': nn_distance, 'nn_key': nn_key, 'nn_distance_threshold': nn_distance_threshold})

                            self.neurons[nn_key]['n_runner_up'] += 1
                            self.neurons[nn_key]['last_runner_up'] = self.update_id

                            # reset the neighbour activation to full strength
                            #
                            self.neurons[nn_key]['activation'] = 1.0

                            # the learning rate for a neighbour needs to be much less that the bmu - hence the product of learning rates and 0.1 factor
                            #
                            nn_learn_rate = self.neurons[bmu_key]['learn_rate'] * self.neurons[nn_key]['learn_rate'] * 0.1

                            # learn the sdr
                            #
                            self.neurons[nn_key]['sdr'].learn(sdr=sdr,
                                                              learn_rate=nn_learn_rate,
                                                              learn_types=learn_types)
                            nn_idx += 1
                            if nn_idx >= len(distances):
                                finished = True
                        else:
                            finished = True

                    # recalculate the distances between updated neurons
                    #
                    nn_processed = set()
                    for neuron_key in updated_neurons:
                        for nn_key in self.neurons[neuron_key]['nn']:
                            pair = (min(neuron_key, nn_key), max(neuron_key, nn_key))
                            if pair not in nn_processed:
                                nn_processed.add(pair)
                                distance = self.neurons[neuron_key]['sdr'].distance(sdr=self.neurons[nn_key]['sdr'],
                                                                                    search_types=search_types)

                                # set the distance
                                #
                                self.neurons[neuron_key]['nn'][nn_key] = distance['distance']
                                self.neurons[nn_key]['nn'][neuron_key] = distance['distance']

            anomaly, motif = self.update_gas_error(bmu_key=bmu_key, bmu_distance=bmu_distance, ref_id=ref_id, new_neuron=por['new_neuron_key'])
            por['anomaly'] = anomaly
            por['motif'] = motif

        por['nos_neurons'] = len(self.neurons)

        return por

    def query(self, sdr, bmu_only: bool = True) -> dict:

        # get the types to search for
        #
        search_types = sdr.get_enc_types()

        # calc the distance of the sdr to the existing neurons
        #
        distances = [(neuron_key,
                      self.neurons[neuron_key]['sdr'].distance(sdr=sdr,
                                                               search_types=search_types),
                      self.neurons[neuron_key]['n_bmu'],
                      self.neurons[neuron_key]['sdr'],
                      self.neurons[neuron_key]['sdr'].get_max_distance(search_types=search_types) * (1 - self.similarity_threshold))
                     for neuron_key in self.neurons]

        # sort in ascending order of distance and descending order of number of times bmu
        #
        distances.sort(key=lambda x: (x[1]['distance'], -x[2]))

        # get closest neuron and all other 'activated neurons'
        #
        activated_neurons = [distances[n_idx]
                             for n_idx in range(len(distances))
                             if n_idx == 0 or distances[n_idx][1]['distance'] <= distances[n_idx][4]]

        por = {'sam': self.name}

        if len(activated_neurons) > 0:

            sum_distance = sum([n[1]['distance'] for n in activated_neurons])

            # select the bmu
            #
            if bmu_only or sum_distance == 0 or len(activated_neurons) == 1:
                por['sdr'] = activated_neurons[0][3]

                if sum_distance > 0 and len(activated_neurons) > 1:
                    por['neurons'] = [{'neuron_key': n[0], 'weight': 1 - (n[1]['distance'] / sum_distance)}
                                      for n in activated_neurons]
                else:
                    por['neurons'] = [{'neuron_key': n[0], 'weight': 1.0} for n in activated_neurons]

            # else create a weighted average of neurons
            #
            else:
                por['sdr'] = SDR()
                por['neurons'] = []
                for n in activated_neurons:
                    weight = 1 - (n[1]['distance'] / sum_distance)
                    por['sdr'].merge(sdr=n[3], weight=weight)
                    por['neurons'].append({'neuron_key': n[0], 'weight': weight})

        return por


if __name__ == '__main__':

    from src.string_encoder import StringEncoder
    from src.numeric_encoder import NumericEncoder

    ng = SAM(name='test',
             similarity_threshold=0.75,
             anomaly_threshold_factor=6.0,
             error_decay=0.1,
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

    t_1 = SDR()
    t_1.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_1.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_1.add_encoding(enc_type='Volume', value=100, encoder=numeric_enc)

    p1 = ng.train(sdr=t_1, ref_id='t_1', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_2 = SDR()
    t_2.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_2.add_encoding(enc_type='Platform', value='B', encoder=str_enc)
    t_2.add_encoding(enc_type='Volume', value=50, encoder=numeric_enc)

    p2 = ng.train(sdr=t_2, ref_id='t_2', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_3 = SDR()
    t_3.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_3.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_3.add_encoding(enc_type='Volume', value=75, encoder=numeric_enc)

    p3 = ng.train(sdr=t_3, ref_id='t_3', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_4 = SDR()
    t_4.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_4.add_encoding(enc_type='Platform', value='B', encoder=str_enc)
    t_4.add_encoding(enc_type='Volume', value=60, encoder=numeric_enc)

    p4 = ng.train(sdr=t_4, ref_id='t_4', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    t_5 = SDR()
    t_5.add_encoding(enc_type='Date', value='22-11-66', encoder=str_enc)
    t_5.add_encoding(enc_type='Platform', value='A', encoder=str_enc)
    t_5.add_encoding(enc_type='Volume', value=110, encoder=numeric_enc)

    p5 = ng.train(sdr=t_5, ref_id='t_5', search_types={'Date', 'Platform', 'Volume'}, learn_types={'Date', 'Platform', 'Volume'})

    ng_dict = ng.to_dict(decode=True)

    t_6 = SDR()
    t_6.add_encoding(enc_type='Volume', value=90, encoder=numeric_enc)

    q_por_bmu = ng.query(sdr=t_6, bmu_only=True)
    q_por_bmu_decode = q_por_bmu['sdr'].decode()
    q_por_weav = ng.query(sdr=t_6, bmu_only=False)
    q_por_weav_decode = q_por_weav['sdr'].decode()

    print('finished')

