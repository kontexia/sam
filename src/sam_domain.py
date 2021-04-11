#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sam import SAM
from src.sgm import SGM


class SAMDomain(object):
    def __init__(self,
                 domain,
                 spatial_search_types,
                 spatial_learn_types,
                 spatial_similarity_threshold: float = 0.75,
                 community_similarity_threshold: float = 0.5,
                 temporal_similarity_threshold: float = 0.75,
                 anomaly_threshold_factor: float = 4.0,
                 similarity_ema_alpha: float = 0.1,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01,
                 ):

        self.domain = domain

        # the fabric of SAMS for this domain
        #
        self.sams = {}

        self.spatial_search_types = spatial_search_types
        self.spatial_learn_types = spatial_learn_types

        self.temporal_search_types = {f'{z}_{enc_type}' for enc_type in spatial_search_types for z in ['z0', 'z1', 'z2']}
        self.temporal_learn_types = {f'{z}_{enc_type}' for enc_type in spatial_learn_types for z in ['z0', 'z1', 'z2']}

        self.spatial_similarity_threshold = spatial_similarity_threshold
        self.community_similarity_threshold = community_similarity_threshold
        self.temporal_similarity_threshold = temporal_similarity_threshold
        self.anomaly_threshold_factor = anomaly_threshold_factor
        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.prune_threshold = prune_threshold

        # short term memory of sparse generalised memories
        #
        self.lstm = []

        self.sams['spatial'] = SAM(name=f'spatial_{domain}',
                                   similarity_threshold=self.spatial_similarity_threshold,
                                   anomaly_threshold_factor=self.anomaly_threshold_factor,
                                   similarity_ema_alpha=self.similarity_ema_alpha,
                                   learn_rate_decay=self.learn_rate_decay,
                                   prune_threshold=self.prune_threshold)

        community_name = f'community_{self.domain}'

        self.sams['community'] = SAM(name=community_name,
                                     similarity_threshold=self.community_similarity_threshold,
                                     anomaly_threshold_factor=self.anomaly_threshold_factor,
                                     similarity_ema_alpha=self.similarity_ema_alpha,
                                     learn_rate_decay=self.learn_rate_decay,
                                     prune_threshold=self.prune_threshold)

        temporal_name = f'temporal_{self.domain}'
        self.sams['temporal'] = SAM(name=temporal_name,
                                    similarity_threshold=self.temporal_similarity_threshold,
                                    anomaly_threshold_factor=self.anomaly_threshold_factor,
                                    similarity_ema_alpha=self.similarity_ema_alpha,
                                    learn_rate_decay=self.learn_rate_decay,
                                    prune_threshold=self.prune_threshold)

    def train(self, sgm: SGM, ref_id: str) -> dict:
        pors = dict()
        pors['spatial'] = self.sams['spatial'].train(sgm=sgm,
                                                     ref_id=ref_id,
                                                     search_types=self.spatial_search_types,
                                                     learn_types=self.spatial_learn_types)

        pors['community'] = self.sams['community'].train(sgm=sgm,
                                                         ref_id=ref_id,
                                                         search_types=self.spatial_search_types,
                                                         learn_types=self.spatial_learn_types)

        # add association sgm to the current long short term memory and train
        #
        temporal_sgm = SGM()
        for idx in range(len(self.lstm)):
            temporal_sgm.merge(sgm=self.lstm[idx], weight=1, enc_type_prefix=f'z{len(self.lstm)-idx}')

        temporal_sgm.merge(sgm=sgm, weight=1.0, enc_type_prefix='z0')

        pors['temporal'] = self.sams['temporal'].train(sgm=temporal_sgm,
                                                       ref_id=ref_id,
                                                       search_types=self.temporal_search_types,
                                                       learn_types=self.temporal_learn_types)

        # now update lstm for next step
        #
        if pors['community']['new_neuron_key'] is not None:
            c_sgm = SGM(sgm=self.sams['community'].neurons[pors['community']['new_neuron_key']]['sgm'])
        else:
            c_sgm = SGM(sgm=self.sams['community'].neurons[pors['community']['bmu_key']]['sgm'])

        self.lstm.append(c_sgm)
        if len(self.lstm) > 2:
            self.lstm.pop(0)

        return pors

    def query_spatial(self, sgm, bmu_only=True) -> dict:

        por = self.sams['spatial'].query(sgm=sgm, bmu_only=bmu_only)
        return por

    def query_community(self, sgm, bmu_only=True) -> dict:

        por = self.sams['community'].query(sgm=sgm, bmu_only=bmu_only)
        return por

    def query_temporal(self, context_sgms, bmu_only=True):

        temporal_sgm = SGM()

        for idx in range(len(context_sgms)):
            temporal_sgm.merge(sgm=context_sgms[idx], weight=1, enc_type_prefix=f'z{len(self.lstm)-idx}')

        por = self.sams['temporal'].query(sgm=temporal_sgm, bmu_only=bmu_only)

        return por


if __name__ == '__main__':
    pass

