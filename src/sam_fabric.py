#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_associative_memory import SAMRegion
from src.sparse_distributed_representation import SDR, ENC_IDX


class SAMFabric(object):
    def __init__(self, association_params: dict):

        self.sams = {}
        self.association_params = association_params
        self.sams['association'] = SAMRegion(name='association',
                                             similarity_threshold=association_params['similarity_threshold'],
                                             community_threshold=association_params['community_threshold'],
                                             temporal_learn_rate=association_params['temporal_learning_rate'],
                                             prune_threshold=association_params['prune_threshold'])

    def to_dict(self, decode: bool = True) -> dict:

        dict_fabric = {}
        for region in self.sams:
            dict_fabric[region] = self.sams[region].to_dict(decode=decode)
        return dict_fabric

    def train(self, region_sdrs, region_sam_params):

        pors = {}

        for region in region_sdrs:

            # create a region sam if required
            #
            if region not in self.sams:
                self.sams[region] = SAMRegion(name=region,
                                              similarity_threshold=region_sam_params[region]['similarity_threshold'],
                                              community_threshold=region_sam_params[region]['community_threshold'],
                                              temporal_learn_rate=region_sam_params[region]['temporal_learning_rate'],
                                              prune_threshold=region_sam_params[region]['prune_threshold'])

            # train this region
            #
            pors[region] = self.sams[region].learn_pattern(sdr=region_sdrs[region],
                                                           activation_enc_keys=region_sam_params[region]['activation_enc_keys'])

        # create an sdr to learn the patterns of association connections
        #
        training_association_sdr = SDR()

        # associate winning neurons with each other
        #
        for region_1 in pors:

            # create an sdr to associate region_1 with all other regions
            #
            region_association_sdr = SDR()
            for region_2 in pors:
                if region_1 != region_2:
                    region_association_sdr.add_encoding(enc_key=(region_2,), encoding={pors[region_2]['activations'][0]['neuron_key']: 1.0})

            # train region_1 associations
            #
            self.sams[region_1].associate(neuron_key=pors[region_1]['activations'][0]['neuron_key'], sdr=region_association_sdr, learn_rate=self.association_params['association_learn_rate'])

            neuron = self.sams[region_1].get_neuron(neuron_key=pors[region_1]['activations'][0]['neuron_key'])

            # collect up the association links
            for sdr_key in neuron['association_sdr'].encoding:

                # create an enc key between region_1 and the region in the sdr
                #
                enc_key = (region_1, sdr_key[ENC_IDX][0])
                training_association_sdr.add_encoding(enc_key=enc_key, encoding=neuron['association_sdr'].encoding[sdr_key])

        pors['association'] = self.sams['association'].learn_pattern(sdr=training_association_sdr)

        return pors
