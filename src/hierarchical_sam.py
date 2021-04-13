#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sam import SAM
from src.sgm import SGM


class HSAM(object):
    def __init__(self,
                 domain,
                 spatial_search_types,
                 spatial_learn_types,
                 spatial_similarity_threshold: float = 0.75,
                 community_threshold_adj: float = 0.1,
                 anomaly_threshold_factor: float = 4.0,
                 similarity_ema_alpha: float = 0.3,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01,
                 ):

        self.domain = domain

        # the fabric of SAMS for this domain
        #
        self.sams = {}

        self.spatial_search_types = spatial_search_types
        self.spatial_learn_types = spatial_learn_types

        self.spatial_similarity_threshold = spatial_similarity_threshold
        self.community_threshold_adj = community_threshold_adj
        self.anomaly_threshold_factor = anomaly_threshold_factor
        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.prune_threshold = prune_threshold

        self.sams['radar'] = SAM(name=f'radar_{domain}',
                                 similarity_threshold=self.spatial_similarity_threshold - self.community_threshold_adj,
                                 community_threshold_adj=self.community_threshold_adj,
                                 anomaly_threshold_factor=self.anomaly_threshold_factor,
                                 similarity_ema_alpha=self.similarity_ema_alpha,
                                 learn_rate_decay=self.learn_rate_decay,
                                 prune_threshold=self.prune_threshold)

    def train(self, sgm: SGM, ref_id: str) -> dict:

        pors = dict()

        # train the radar sam
        #
        pors['radar'] = self.sams['radar'].train(sgm=sgm,
                                                 ref_id=ref_id,
                                                 search_types=self.spatial_search_types,
                                                 learn_types=self.spatial_learn_types)

        if pors['radar']['new_neuron_key'] is not None:
            neuron_key = pors['radar']['new_neuron_key']
            self.sams[neuron_key] = SAM(name=f'{neuron_key}_{self.domain}',
                                        similarity_threshold=self.spatial_similarity_threshold,
                                        community_threshold_adj=self.community_threshold_adj,
                                        anomaly_threshold_factor=self.anomaly_threshold_factor,
                                        similarity_ema_alpha=self.similarity_ema_alpha,
                                        learn_rate_decay=self.learn_rate_decay,
                                        prune_threshold=self.prune_threshold)
        else:
            neuron_key = pors['radar']['bmu_key']

        pors[neuron_key] = self.sams[neuron_key].train(sgm=sgm,
                                                       ref_id=ref_id,
                                                       search_types=self.spatial_search_types,
                                                       learn_types=self.spatial_learn_types)

        return pors

    def query_spatial(self, sgm, bmu_only=True) -> dict:

        radar_por = self.sams['radar'].query(sgm=sgm, bmu_only=True)

        neuron_key = radar_por['neurons'][0]['neuron_key']
        por = self.sams[neuron_key].query(sgm=sgm, bmu_only=bmu_only)

        return por
