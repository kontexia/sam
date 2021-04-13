#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sam import SAM
from src.sgm import SGM


class HSAM(object):
    def __init__(self,
                 domain,
                 search_types,
                 learn_types,
                 layer_1_similarity_threshold: float = 0.5,
                 layer_1_community_threshold: float = 0.4,
                 layer_2_similarity_threshold: float = 0.7,
                 layer_2_community_threshold: float = 0.6,
                 anomaly_threshold_factor: float = 4.0,
                 similarity_ema_alpha: float = 0.3,
                 learn_rate_decay: float = 0.3,
                 prune_threshold: float = 0.01,
                 ):

        self.domain = domain

        # the fabric of SAMS for this domain
        #
        self.sams = {}

        self.search_types = search_types
        self.learn_types = learn_types

        self.layer_1_similarity_threshold = layer_1_similarity_threshold
        self.layer_1_community_threshold = layer_1_community_threshold
        self.layer_2_similarity_threshold = layer_2_similarity_threshold
        self.layer_2_community_threshold = layer_2_community_threshold

        self.anomaly_threshold_factor = anomaly_threshold_factor
        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.prune_threshold = prune_threshold

        self.sams['layer_1'] = SAM(name=f'layer1_{domain}',
                                   similarity_threshold=self.layer_1_similarity_threshold,
                                   community_threshold=self.layer_1_community_threshold,
                                   anomaly_threshold_factor=self.anomaly_threshold_factor,
                                   similarity_ema_alpha=self.similarity_ema_alpha,
                                   learn_rate_decay=self.learn_rate_decay,
                                   prune_threshold=self.prune_threshold)

    def train(self, sgm: SGM, ref_id: str) -> dict:

        pors = dict()

        # train the radar sam
        #
        pors['layer_1'] = self.sams['layer_1'].train(sgm=sgm,
                                                     ref_id=ref_id,
                                                     search_types=self.search_types,
                                                     learn_types=self.learn_types)

        if pors['layer_1']['new_neuron_key'] is not None:
            neuron_key = pors['layer_1']['new_neuron_key']
            self.sams[neuron_key] = SAM(name=f'{neuron_key}_{self.domain}',
                                        similarity_threshold=self.layer_2_similarity_threshold,
                                        community_threshold=self.layer_2_community_threshold,
                                        anomaly_threshold_factor=self.anomaly_threshold_factor,
                                        similarity_ema_alpha=self.similarity_ema_alpha,
                                        learn_rate_decay=self.learn_rate_decay,
                                        prune_threshold=self.prune_threshold)
        else:
            neuron_key = pors['layer_1']['bmu_key']

        pors[neuron_key] = self.sams[neuron_key].train(sgm=sgm,
                                                       ref_id=ref_id,
                                                       search_types=self.search_types,
                                                       learn_types=self.learn_types)

        return pors

    def query(self, sgm, bmu_only=True) -> dict:

        layer_1_por = self.sams['layer_1'].query(sgm=sgm, bmu_only=True)

        neuron_key = layer_1_por['neurons'][0]['neuron_key']
        por = self.sams[neuron_key].query(sgm=sgm, bmu_only=bmu_only)

        return por
