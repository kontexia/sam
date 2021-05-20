#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.sparse_associative_memory import SAM
from src.sparse_distributed_representation import SDR, ENC_IDX
from typing import Union


class SAMFabric(object):
    """
    SAMFabric defines a collection of sam regions which are trained together and related through association connections

    """

    def __init__(self, association_params: dict):

        # a map fo regions to sams
        #
        self.sams = {}

        # the parameters for the association sam
        #
        self.association_params = association_params

        # instantiate the association sam
        #
        self.sams['association'] = SAM(name='association',
                                       similarity_threshold=association_params['similarity_threshold'],
                                       community_factor=association_params['community_factor'],
                                       temporal_learn_rate=association_params['temporal_learning_rate'],
                                       prune_threshold=association_params['prune_threshold'])

    def to_dict(self, decode: bool = True) -> dict:

        dict_fabric = {'association_params': {param: self.association_params[param]
                                              for param in self.association_params}}
        for region in self.sams:
            dict_fabric[region] = self.sams[region].to_dict(decode=decode)
        return dict_fabric

    def train(self, region_sdrs: dict, region_sam_params: dict):

        pors = {}

        for region in region_sdrs:

            # create a region sam if required
            #
            if region not in self.sams:
                self.sams[region] = SAM(name=region,
                                        similarity_threshold=region_sam_params[region]['similarity_threshold'],
                                        community_factor=region_sam_params[region]['community_factor'],
                                        temporal_learn_rate=region_sam_params[region]['temporal_learning_rate'],
                                        prune_threshold=region_sam_params[region]['prune_threshold'])

            # train this region
            #
            pors[region] = self.sams[region].learn_pattern(sdr=region_sdrs[region],
                                                           activation_enc_keys=region_sam_params[region]['activation_enc_keys'])

        # an sdr to learn the patterns of association connections between the bmu neurons in each region
        #
        training_association_sdr = SDR()

        # associate winning neurons with each other
        #
        for region_1 in pors:

            region_1_neuron_key = pors[region_1]['activations'][0]['neuron_key']

            # create an sdr to associate region_1 with all other regions
            #
            region_association_sdr = SDR()
            for region_2 in pors:
                if region_1 != region_2:
                    region_association_sdr.add_encoding(enc_key=(region_2,), encoding={pors[region_2]['activations'][0]['neuron_key']: 1.0})

            # train region_1 bmu neuron's associations
            #
            self.sams[region_1].associate(neuron_key=region_1_neuron_key, sdr=region_association_sdr, learn_rate=self.association_params['association_learn_rate'])

            neuron = self.sams[region_1].get_neuron(neuron_key=region_1_neuron_key)

            # collect up the association links
            #
            for sdr_key in neuron['association_sdr']['encoding']:

                # create an enc key between region_1 neuron and the region in the sdr
                #
                enc_key = (region_1, region_1_neuron_key, sdr_key[ENC_IDX][0])
                training_association_sdr.add_encoding(enc_key=enc_key, encoding=neuron['association_sdr']['encoding'][sdr_key])

        pors['association'] = self.sams['association'].learn_pattern(sdr=training_association_sdr)

        return pors

    def query(self, region_sdr: Union[list, dict], similarity_threshold: float = None, decode: bool = True) -> dict:
        pors = {}
        if isinstance(region_sdr, list):

            # prepare the lists of sdrs for each region
            #
            region_sdrs = {}
            for item in region_sdr:
                for region in item:
                    if region not in region_sdrs:
                        region_sdrs[region] = [item[region]]
                    else:
                        region_sdrs[region].append(item[region])

            for region in region_sdrs:
                pors[region] = self.sams[region].query(sdr=region_sdrs[region],
                                                       similarity_threshold=similarity_threshold,
                                                       decode=decode)

        else:
            for region in region_sdr:
                pors[region] = self.sams[region].query(sdr=region_sdr[region],
                                                       similarity_threshold=similarity_threshold,
                                                       decode=decode)

        activated_regions = {}

        for region in pors:
            if region not in activated_regions:
                activated_regions[region] = {}

            if pors[region]['bmu_key'] not in activated_regions[region]:
                activated_regions[region][pors[region]['bmu_key']] = pors[region]['activations'][0]['similarity']
            else:
                activated_regions[region][pors[region]['bmu_key']] += pors[region]['activations'][0]['similarity']

            bmu_neuron = self.sams[region].get_neuron(neuron_key=pors[region]['bmu_key'])

            for sdr_key in bmu_neuron['association_sdr']['encoding']:
                if sdr_key[ENC_IDX][0] not in activated_regions:
                    activated_regions[sdr_key[ENC_IDX][0]] = {}
                for neuron_key in bmu_neuron['association_sdr']['encoding'][sdr_key]:
                    if neuron_key not in activated_regions[sdr_key[ENC_IDX][0]]:
                        activated_regions[sdr_key[ENC_IDX][0]][neuron_key] = bmu_neuron['association_sdr']['encoding'][sdr_key][neuron_key]
                    else:
                        activated_regions[sdr_key[ENC_IDX][0]][neuron_key] += bmu_neuron['association_sdr']['encoding'][sdr_key][neuron_key]

        for region in activated_regions:
            if region not in pors:
                associated_region = [{'neuron_key': neuron_key,
                                      'activation': activated_regions[region][neuron_key],
                                      'sdr': self.sams[region].get_neuron(neuron_key=neuron_key, decode=decode)['pattern_sdr']}
                                     for neuron_key in activated_regions[region]]
                if len(associated_region) > 0:
                    associated_region.sort(key=lambda x: x['activation'], reverse=True)
                    pors[region] = {'bmu_key': associated_region[0]['neuron_key'],
                                    'sdr': associated_region[0]['sdr'],
                                    'activations': associated_region}

        return pors


