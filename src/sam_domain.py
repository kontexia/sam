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
                 community_threshold_adj: float = 0.05,
                 temporal_similarity_threshold: float = 0.75,
                 temporal_memory: int = 10,
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
        self.temporal_similarity_threshold = temporal_similarity_threshold
        self.anomaly_threshold_factor = anomaly_threshold_factor
        self.similarity_ema_alpha = similarity_ema_alpha
        self.learn_rate_decay = learn_rate_decay
        self.prune_threshold = prune_threshold

        # window of sparse generalised memories
        #
        self.temporal_memory = []

        # the max length allowed for the window
        #
        self.temporal_memory_window = temporal_memory

        # the search and learn types with temporal prefixes
        #
        self.temporal_search_types = {f'z{z}_{enc_type}' for enc_type in spatial_search_types for z in range(self.temporal_memory_window)}
        self.temporal_learn_types = {f'z{z}_{enc_type}' for enc_type in spatial_learn_types for z in range(self.temporal_memory_window)}

        self.sams['spatial'] = SAM(name=f'spatial_{domain}',
                                   similarity_threshold=self.spatial_similarity_threshold,
                                   community_threshold_adj=self.community_threshold_adj,
                                   anomaly_threshold_factor=self.anomaly_threshold_factor,
                                   similarity_ema_alpha=self.similarity_ema_alpha,
                                   learn_rate_decay=self.learn_rate_decay,
                                   prune_threshold=self.prune_threshold)

        temporal_name = f'temporal_{self.domain}'
        self.sams['temporal'] = SAM(name=temporal_name,
                                    similarity_threshold=self.temporal_similarity_threshold,
                                    community_threshold_adj=self.community_threshold_adj,
                                    anomaly_threshold_factor=self.anomaly_threshold_factor,
                                    similarity_ema_alpha=self.similarity_ema_alpha,
                                    learn_rate_decay=self.learn_rate_decay,
                                    prune_threshold=self.prune_threshold)

    def train(self, sgm: SGM, ref_id: str) -> dict:

        pors = dict()

        # train the spatial sam
        #
        pors['spatial'] = self.sams['spatial'].train(sgm=sgm,
                                                     ref_id=ref_id,
                                                     search_types=self.spatial_search_types,
                                                     learn_types=self.spatial_learn_types)

        # train temporal memory if required
        #
        if self.temporal_memory_window > 0:

            # add incoming data into the window and ensure its max size is kept constant
            #
            self.temporal_memory.append(sgm)
            if len(self.temporal_memory) > self.temporal_memory_window:
                self.temporal_memory.pop(0)

            # create temporal memory sgm from window of raw sgms
            #
            temporal_sgm = SGM()
            for idx in range(len(self.temporal_memory)):
                # each raw sgm has a prefix related to its index in the window
                #
                temporal_sgm.merge(sgm=self.temporal_memory[idx], weight=1, enc_type_prefix=f'z{len(self.temporal_memory)-idx-1}')

            pors['temporal'] = self.sams['temporal'].train(sgm=temporal_sgm,
                                                           ref_id=ref_id,
                                                           search_types=self.temporal_search_types,
                                                           learn_types=self.temporal_learn_types)

        return pors

    def query_spatial(self, sgm, bmu_only=True) -> dict:

        por = self.sams['spatial'].query(sgm=sgm, bmu_only=bmu_only)
        return por

    def query_temporal(self, context_sgms, bmu_only=True):

        # assume context_sgms is a list (window) of contextual sgms, starting with oldest
        #
        # create an sgm with prefixes related to position in window
        temporal_sgm = SGM()
        for idx in range(len(context_sgms)):
            temporal_sgm.merge(sgm=context_sgms[idx], weight=1, enc_type_prefix=f'z{len(context_sgms)-idx}')

        por = self.sams['temporal'].query(sgm=temporal_sgm, bmu_only=bmu_only)

        return por


if __name__ == '__main__':
    pass

