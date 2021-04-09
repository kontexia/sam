#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sam import SAM
from src.sgm import SGM


class SAMFabric(object):
    def __init__(self,
                 domain,
                 assoc_anomaly_threshold_factor: float = 4.0,
                 assoc_similarity_threshold: float = 0.75,
                 temporal_anomaly_threshold_factor: float = 4.0,
                 temporal_similarity_threshold: float = 0.75,
                 similarity_ema_alpha: float = 0.1,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01,
                 prune_neurons: bool = False,
                 ):

        self.domain = domain

        # the fabric of SAMS
        #
        self.sams = {}

        # long short term memory of previous sam neurons that were activated
        #
        self.lstm = SGM()

        self.assoc_anomaly_threshold_factor = assoc_anomaly_threshold_factor
        self.assoc_similarity_threshold = assoc_similarity_threshold

        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.prune_threshold = prune_threshold
        self.prune_neurons = prune_neurons

        self.temporal_anomaly_threshold_factor = temporal_anomaly_threshold_factor
        self.temporal_similarity_threshold = temporal_similarity_threshold

        # fast decay
        #
        self.temporal_term_1_decay = 0.7

        # slow decay
        #
        self.temporal_term_2_decay = 0.3

    def create_sam(self,
                   name,
                   similarity_threshold=0.75,
                   anomaly_threshold_factor: float = 4.0,
                   search_types: set = None,
                   learn_types: set = None,
                   create_temporal: bool = False):
        if name not in self.sams:
            self.sams[name] = {'sam': SAM(name=name,
                                          similarity_threshold=similarity_threshold,
                                          anomaly_threshold_factor=anomaly_threshold_factor,
                                          similarity_ema_alpha=self.similarity_ema_alpha,
                                          learn_rate_decay=self.learn_rate_decay,
                                          prune_threshold=self.prune_threshold,
                                          prune_neurons=self.prune_neurons),
                               'search_types': search_types,
                               'learn_types': learn_types
                               }

            assoc_name = f'association_{self.domain}'

            if len(self.sams) > 1:
                assoc_search_types = {f'{name}_{s_type}' for s_type in search_types}
                assoc_learn_types = {f'{name}_{l_type}' for l_type in learn_types}

                if assoc_name not in self.sams:
                    self.sams[assoc_name] = {'sam': SAM(name=assoc_name,
                                                        similarity_threshold=self.assoc_similarity_threshold,
                                                        anomaly_threshold_factor=self.assoc_anomaly_threshold_factor,
                                                        similarity_ema_alpha=self.similarity_ema_alpha,
                                                        learn_rate_decay=self.learn_rate_decay,
                                                        prune_threshold=self.prune_threshold,
                                                        prune_neurons=self.prune_neurons),
                                             'search_types': assoc_search_types,
                                             'learn_types': assoc_learn_types
                                             }
                    for exist_name in self.sams:
                        if exist_name != name and exist_name != assoc_name:
                            self.sams[assoc_name]['search_types'].update(assoc_search_types)
                            self.sams[assoc_name]['learn_types'].update(assoc_learn_types)

                else:
                    self.sams[assoc_name]['search_types'].update(assoc_search_types)
                    self.sams[assoc_name]['learn_types'].update(assoc_learn_types)

            if create_temporal:
                temporal_name = f'temporal_{self.domain}'
                if assoc_name in self.sams:
                    temporal_search_types = {f'{t}_{s_type}' for s_type in self.sams[assoc_name]['search_types'] for t in ['z0', 'z1', 'z2']}
                    temporal_learn_types = {f'{t}_{l_type}' for l_type in self.sams[assoc_name]['learn_types'] for t in ['z0', 'z1', 'z2']}
                else:
                    temporal_search_types = {f'{t}_{s_type}' for s_type in search_types for t in ['z0', 'z1', 'z2']}
                    temporal_learn_types = {f'{t}_{l_type}' for l_type in learn_types for t in ['z0', 'z1', 'z2']}

                if temporal_name not in self.sams:
                    self.sams[temporal_name] = {'sam': SAM(name=temporal_name,
                                                           similarity_threshold=self.temporal_similarity_threshold,
                                                           anomaly_threshold_factor=self.temporal_anomaly_threshold_factor,
                                                           similarity_ema_alpha=self.similarity_ema_alpha,
                                                           learn_rate_decay=self.learn_rate_decay,
                                                           prune_threshold=self.prune_threshold,
                                                           prune_neurons=self.prune_neurons,
                                                           ),
                                                'search_types': temporal_search_types,
                                                'learn_types': temporal_learn_types
                                                }
                else:
                    self.sams[temporal_name]['search_types'].update(temporal_search_types)
                    self.sams[temporal_name]['learn_types'].update(temporal_learn_types)

    def train(self, sgms, ref_id):
        pors = {}
        for name in sgms:
            if name in self.sams:
                por = self.sams[name]['sam'].train(sgm=sgms[name],
                                                   ref_id=ref_id,
                                                   search_types=self.sams[name]['search_types'],
                                                   learn_types=self.sams[name]['learn_types'])
                pors[name] = por

        if len(sgms) > 1:

            # train the association sam
            # create an sgm containing entries from each sam activated neuron
            #
            assoc_name = f'association_{self.domain}'

            assoc_train_sgm = SGM()

            for name in pors:
                if pors[name]['new_neuron_key'] is not None:
                    neuron_to_merge = pors[name]['new_neuron_key']
                else:
                    neuron_to_merge = pors[name]['bmu_key']

                assoc_train_sgm.merge(sgm=self.sams[name]['sam'].neurons[neuron_to_merge]['sgm'], weight=1.0, enc_type_prefix=name)

            por = self.sams[assoc_name]['sam'].train(sgm=assoc_train_sgm,
                                                     ref_id=ref_id,
                                                     search_types=self.sams[assoc_name]['search_types'],
                                                     learn_types=self.sams[assoc_name]['learn_types'])
            pors[assoc_name] = por

            # get the bmu from the association sam
            #
            if pors[assoc_name]['new_neuron_key'] is not None:
                neuron_to_merge = pors[assoc_name]['new_neuron_key']
            else:
                neuron_to_merge = pors[assoc_name]['bmu_key']

            assoc_sgm = self.sams[assoc_name]['sam'].neurons[neuron_to_merge]['sgm']

        # else only one name in the sgm
        #
        else:

            name = list(sgms.keys())[0]

            if pors[name]['new_neuron_key'] is not None:
                neuron_to_merge = pors[name]['new_neuron_key']
            else:
                neuron_to_merge = pors[name]['bmu_key']

            assoc_sgm = self.sams[name]['sam'].neurons[neuron_to_merge]['sgm']

        # now train the temporal sam if it exists
        #
        temporal_name = f'temporal_{self.domain}'
        if temporal_name in self.sams:
            temporal_train_sgm = SGM(sgm=self.lstm)
            temporal_train_sgm.merge(sgm=assoc_sgm, weight=1.0, enc_type_prefix='z0')

            por = self.sams[temporal_name]['sam'].train(sgm=temporal_train_sgm,
                                                        ref_id=ref_id,
                                                        search_types=self.sams[temporal_name]['search_types'],
                                                        learn_types=self.sams[temporal_name]['learn_types'])
            pors[temporal_name] = por

            # now update lstm for next step
            #
            temporal_sgm = SGM()
            temporal_sgm.copy(sgm=assoc_sgm, enc_type_prefix='z1')
            self.lstm.learn(sgm=temporal_sgm,
                            learn_rate=self.temporal_term_1_decay,
                            learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z1' in l_type},
                            prune=self.prune_threshold)

            temporal_sgm = SGM()
            temporal_sgm.copy(sgm=assoc_sgm, enc_type_prefix='z2')
            self.lstm.learn(sgm=temporal_sgm,
                            learn_rate=self.temporal_term_2_decay,
                            learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z2' in l_type},
                            prune=self.prune_threshold)

        return pors

    def query_association(self, sgms, bmu_only=True):

        por = {}

        assoc_name = f'association_{self.domain}'
        if assoc_name in self.sams:
            q_sgm = SGM()
            for name in sgms:
                q_sgm.merge(sgm=sgms[name], weight=1.0, enc_type_prefix=name)

            por = self.sams[assoc_name]['sam'].query(sgm=q_sgm, bmu_only=True)

        # if no association then just query using the first sgm in sgms
        else:
            for name in sgms:
                if name in self.sams:
                    por = self.sams[name]['sam'].query(sgm=sgms[name], bmu_only=bmu_only)
                    break

        return por

    def query_temporal(self, context_sdrs, bmu_only=True):

        pors = {}

        temporal_name = f'temporal_{self.domain}'

        if temporal_name in self.sams:

            # will need to create a lstm of the query results
            #
            lstm = SGM()
            for idx in range(len(context_sdrs)):
                pors[idx] = self.query_association(sgms=context_sdrs[idx], bmu_only=bmu_only)

                assoc_sgm = pors[idx]['sgm']

                # now update lstm for next step
                #
                temporal_sgm = SGM()
                temporal_sgm.copy(sgm=assoc_sgm, enc_type_prefix='z1')
                lstm.learn(sgm=temporal_sgm,
                           learn_rate=self.temporal_term_1_decay,
                           learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z1' in l_type},
                           prune=self.prune_threshold)

                temporal_sgm = SGM()
                temporal_sgm.copy(sgm=assoc_sgm, enc_type_prefix='z2')
                lstm.learn(sgm=temporal_sgm,
                           learn_rate=self.temporal_term_2_decay,
                           learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z2' in l_type},
                           prune=self.prune_threshold)

            pors[temporal_name] = self.sams[temporal_name]['sam'].query(sgm=lstm, bmu_only=bmu_only)

        return pors


if __name__ == '__main__':
    pass

