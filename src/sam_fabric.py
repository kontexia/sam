#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sam import SAM
from src.sdr import SDR


class SAMFabric(object):
    def __init__(self,
                 domain,
                 assoc_anomaly_threshold_factor: float = 4.0,
                 assoc_error_decay: float = 0.1,
                 assoc_similarity_threshold: float = 0.75,
                 assoc_learn_rate_decay: float = 0.3,
                 assoc_prune_threshold: float = 0.01,
                 assoc_prune_neurons: bool = False,
                 temporal_anomaly_threshold_factor: float = 4.0,
                 temporal_error_decay: float = 0.1,
                 temporal_similarity_threshold: float = 0.75,
                 temporal_learn_rate_decay: float = 0.3,
                 temporal_prune_threshold: float = 0.01,
                 temporal_prune_neurons: bool = False):

        self.domain = domain

        # the fabric of SAMS
        #
        self.sams = {}

        # long short term memory of previous sam neurons that were activated
        #
        self.lstm = SDR()

        self.assoc_anomaly_threshold_factor = assoc_anomaly_threshold_factor
        self.assoc_error_decay = assoc_error_decay
        self.assoc_learn_rate_decay = assoc_learn_rate_decay
        self.assoc_prune_threshold = assoc_prune_threshold
        self.assoc_prune_neurons = assoc_prune_neurons
        self.assoc_similarity_threshold = assoc_similarity_threshold

        self.temporal_anomaly_threshold_factor = temporal_anomaly_threshold_factor
        self.temporal_error_decay = temporal_error_decay
        self.temporal_learn_rate_decay = temporal_learn_rate_decay
        self.temporal_prune_threshold = temporal_prune_threshold
        self.temporal_prune_neurons = temporal_prune_neurons
        self.temporal_similarity_threshold = temporal_similarity_threshold
        self.temporal_term_1_decay = temporal_learn_rate_decay
        self.temporal_term_2_decay = 1 - temporal_learn_rate_decay

    def create_sam(self,
                   name,
                   similarity_threshold=0.75,
                   anomaly_threshold_factor: float = 4.0,
                   error_decay: float = 0.1,
                   learn_rate_decay: float = 0.3,
                   prune_threshold: float = 0.01,
                   prune_neurons: bool = False,
                   search_types: set = None,
                   learn_types: set = None):
        if name not in self.sams:
            self.sams[name] = {'sam': SAM(name=name,
                                          similarity_threshold=similarity_threshold,
                                          anomaly_threshold_factor=anomaly_threshold_factor,
                                          error_decay=error_decay,
                                          learn_rate_decay=learn_rate_decay,
                                          prune_threshold=prune_threshold,
                                          prune_neurons=prune_neurons),
                               'search_types': search_types,
                               'learn_types': learn_types
                               }

            if len(self.sams) > 1:
                assoc_name = f'{self.domain}_association'
                if assoc_name not in self.sams:
                    self.sams[assoc_name] = {'sam': SAM(name=assoc_name,
                                                        similarity_threshold=self.assoc_similarity_threshold,
                                                        anomaly_threshold_factor=self.assoc_anomaly_threshold_factor,
                                                        error_decay=self.assoc_error_decay,
                                                        learn_rate_decay=self.assoc_learn_rate_decay,
                                                        prune_threshold=self.assoc_prune_threshold,
                                                        prune_neurons=self.assoc_prune_neurons),
                                             'search_types': search_types,
                                             'learn_types': learn_types
                                             }
                    for exist_name in self.sams:
                        if exist_name != name and exist_name != assoc_name:
                            self.sams[assoc_name]['search_types'].update(search_types)
                            self.sams[assoc_name]['learn_types'].update(learn_types)

                else:
                    self.sams[assoc_name]['search_types'].update(search_types)
                    self.sams[assoc_name]['learn_types'].update(learn_types)

            temporal_name = f'{self.domain}_temporal'
            if temporal_name not in self.sams:
                self.sams[temporal_name] = {'sam': SAM(name=temporal_name,
                                                       similarity_threshold=self.temporal_similarity_threshold,
                                                       anomaly_threshold_factor=self.temporal_anomaly_threshold_factor,
                                                       error_decay=self.temporal_error_decay,
                                                       learn_rate_decay=self.temporal_learn_rate_decay,
                                                       prune_threshold=self.temporal_prune_threshold,
                                                       prune_neurons=self.temporal_prune_neurons,
                                                       ),
                                            'search_types': {f'{s_type}_{t}' for s_type in search_types for t in ['z0', 'z1', 'z2']},
                                            'learn_types': {f'{l_type}_{t}' for l_type in learn_types for t in ['z0', 'z1', 'z2']}
                                            }
            else:
                self.sams[temporal_name]['search_types'].update({f'{s_type}_{t}' for s_type in search_types for t in ['z0', 'z1', 'z2']})
                self.sams[temporal_name]['learn_types'].update({f'{l_type}_{t}' for l_type in learn_types for t in ['z0', 'z1', 'z2']})

    def train(self, sdrs, ref_id):
        pors = {}
        for name in sdrs:
            if name in self.sams:
                por = self.sams[name]['sam'].train(sdr=sdrs[name],
                                                   ref_id=ref_id,
                                                   search_types=self.sams[name]['search_types'],
                                                   learn_types=self.sams[name]['learn_types'])
                pors[name] = por

        assoc_sdr = SDR()
        if len(sdrs) > 1:

            # train the association sam
            # create an sdr containing entries from each sam activated neuron
            #
            assoc_name = f'{self.domain}_association'

            assoc_train_sdr = SDR()

            for name in pors:
                if pors[name]['new_neuron_key'] is not None:
                    neuron_to_merge = pors[name]['new_neuron_key']
                else:
                    neuron_to_merge = pors[name]['bmu_key']

                assoc_train_sdr.merge(sdr=self.sams[name]['sam'].neurons[neuron_to_merge]['sdr'], weight=1.0, enc_type_postfix=name)

            por = self.sams[assoc_name]['sam'].train(sdr=assoc_train_sdr,
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

            assoc_sdr.merge(sdr=self.sams[assoc_name]['sam'].neurons[neuron_to_merge]['sdr'], weight=1.0, enc_type_postfix=None)

        else:
            for name in pors:

                if pors[name]['new_neuron_key'] is not None:
                    neuron_to_merge = pors[name]['new_neuron_key']
                else:
                    neuron_to_merge = pors[name]['bmu_key']

                assoc_sdr.merge(sdr=self.sams[name]['sam'].neurons[neuron_to_merge]['sdr'], weight=1.0, enc_type_postfix=None)

        # now train the temporal sam
        #
        temporal_name = f'{self.domain}_temporal'

        temporal_train_sdr = SDR(sdr=self.lstm)
        temporal_train_sdr.merge(sdr=assoc_sdr, weight=1.0, enc_type_postfix='z0')

        por = self.sams[temporal_name]['sam'].train(sdr=temporal_train_sdr,
                                                    ref_id=ref_id,
                                                    search_types=self.sams[temporal_name]['search_types'],
                                                    learn_types=self.sams[temporal_name]['learn_types'])
        pors[temporal_name] = por

        # now update lstm for next step
        #
        temporal_sdr = SDR()
        temporal_sdr.merge(sdr=assoc_sdr, weight=1.0, enc_type_postfix='z1')
        self.lstm.learn(sdr=temporal_sdr,
                        learn_rate=self.temporal_term_1_decay,
                        learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if '_z1' in l_type},
                        prune=self.temporal_prune_threshold)

        temporal_sdr = SDR()
        temporal_sdr.merge(sdr=assoc_sdr, weight=1.0, enc_type_postfix='z2')
        self.lstm.learn(sdr=temporal_sdr,
                        learn_rate=self.temporal_term_2_decay,
                        learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if '_z2' in l_type},
                        prune=self.temporal_prune_threshold)

        return pors

    def query_association(self, sdrs):

        pors = {}

        last_name = None
        for name in sdrs:
            if name in self.sams:
                last_name = name

                por = self.sams[name]['sam'].query(sdr=sdrs[name], bmu_only=True)
                pors[name] = por

        assoc_name = f'{self.domain}_association'

        if len(pors) > 1:
            assoc_query_sdr = SDR()

            for name in pors:
                if pors[name]['new_neuron_key'] is not None:
                    neuron_to_merge = pors[name]['new_neuron_key']
                else:
                    neuron_to_merge = pors[name]['bmu_key']

                assoc_query_sdr.merge(sdr=self.sams[name]['sam'].neurons[neuron_to_merge]['sdr'], weight=1.0, enc_type_postfix=name)

            por = self.sams[assoc_name]['sam'].query(sdr=assoc_query_sdr, bmu_only=True)
            pors[assoc_name] = por
        else:
            # point the association to the last named data
            #
            pors[assoc_name] = pors[last_name]
        return pors

    def query_context(self, context_sdrs):

        pors = {}

        assoc_name = f'{self.domain}_association'
        temporal_name = f'{self.domain}_temporal'

        # will need to create a lstm of the query results
        #
        lstm = SDR()
        for idx in range(len(context_sdrs)):
            pors[idx] = self.query_association(sdrs=context_sdrs[idx])

            assoc_sdr = pors[idx][assoc_name]['sdr']

            # now update lstm for next step
            #
            temporal_sdr = SDR()
            temporal_sdr.merge(sdr=assoc_sdr, weight=1.0, enc_type_postfix='z1')
            lstm.learn(sdr=temporal_sdr,
                       learn_rate=self.temporal_term_1_decay,
                       learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z1' in l_type},
                       prune=self.temporal_prune_threshold)

            temporal_sdr = SDR()
            temporal_sdr.merge(sdr=assoc_sdr, weight=1.0, enc_type_postfix='z2')
            lstm.learn(sdr=temporal_sdr,
                       learn_rate=self.temporal_term_2_decay,
                       learn_types={l_type for l_type in self.sams[temporal_name]['learn_types'] if 'z2' in l_type},
                       prune=self.temporal_prune_threshold)

        pors[temporal_name] = self.query_association(sdrs={temporal_name: lstm})[assoc_name]

        return pors


if __name__ == '__main__':
    pass

