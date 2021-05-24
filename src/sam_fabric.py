#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.spatial_pooler import SpatialPooler
from src.sparse_distributed_representation import SDR, ENC_IDX, TEMPORAL_IDX
from typing import Union


class SAM(object):
    """
    SAMFabric defines a collection of sam regions which are trained together and related through association connections

    """

    def __init__(self, spatial_params: dict, association_params: dict = None, temporal_params: dict = None):

        # a map of poolers
        #
        self.poolers = {}

        # the parameters for the spatial poolers
        #
        self.spatial_params = spatial_params

        for region in self.spatial_params:
            if 'similarity_threshold' not in self.spatial_params[region]:
                self.spatial_params[region]['similarity_threshold'] = None
            if 'community_factor' not in self.spatial_params[region]:
                self.spatial_params[region]['community_factor'] = None
            if 'prune_threshold' not in self.spatial_params[region]:
                self.spatial_params[region]['prune_threshold'] = None
            if 'activation_enc_keys' not in self.spatial_params[region]:
                self.spatial_params[region]['activation_enc_keys'] = None

        # the parameters for the association pooler
        #
        self.association_params = association_params

        # the parameters for the temporal pooler
        #
        self.temporal_params = temporal_params

        # instantiate the association pooler if required
        #
        if self.association_params is not None:
            if 'similarity_threshold' not in self.association_params:
                self.association_params['similarity_threshold'] = None
            if 'community_factor' not in self.association_params:
                self.association_params['community_factor'] = None
            if 'prune_threshold' not in self.association_params:
                self.association_params['prune_threshold'] = None

            self.poolers['_association'] = SpatialPooler(name='_association',
                                                         similarity_threshold=self.association_params['similarity_threshold'],
                                                         community_factor=self.association_params['community_factor'],
                                                         prune_threshold=self.association_params['prune_threshold'])

        # instantiate the temporal pooler if required
        #
        if self.temporal_params is not None:
            if 'similarity_threshold' not in self.temporal_params:
                self.temporal_params['similarity_threshold'] = None
            if 'community_factor' not in self.temporal_params:
                self.temporal_params['community_factor'] = None
            if 'prune_threshold' not in self.temporal_params:
                self.temporal_params['prune_threshold'] = None
            if 'lstm_learn_rate' not in self.temporal_params:
                self.temporal_params['lstm_learn_rate'] = 0.6

            self.poolers['_temporal'] = SpatialPooler(name='_temporal',
                                                      similarity_threshold=self.temporal_params['similarity_threshold'],
                                                      community_factor=self.temporal_params['community_factor'],
                                                      prune_threshold=self.temporal_params['prune_threshold'])
            self.lstm_sdr = SDR()
        else:
            self.lstm_sdr = None

    def to_dict(self, decode: bool = True) -> dict:

        dict_fabric = {'association_params': self.association_params,
                       'temporal_params': self.temporal_params,
                       'lstm_sdr': None}

        if self.lstm_sdr is not None:
            dict_fabric['lstm_sdr'] = self.lstm_sdr.to_dict(decode=decode)

        for pooler in self.poolers:
            dict_fabric[pooler] = self.poolers[pooler].to_dict(decode=decode)

        return dict_fabric

    def train(self, sdrs: dict):

        pors = {}

        # assume the key in sdrs identifies a region in the fabric
        #
        for region in sdrs:

            if region not in self.spatial_params:
                config = 'default'
            else:
                config = region

            # create a spatial pooler for region if required
            #
            if region not in self.poolers:
                self.poolers[region] = SpatialPooler(name=region,
                                                     similarity_threshold=self.spatial_params[config]['similarity_threshold'],
                                                     community_factor=self.spatial_params[config]['community_factor'],
                                                     prune_threshold=self.spatial_params[config]['prune_threshold'])

            # train this region
            #
            pors[region] = self.poolers[region].learn_pattern(sdr=sdrs[region],
                                                              activation_enc_keys=self.spatial_params[config]['activation_enc_keys'])

        # train the association pooler if required
        #
        if '_association' in self.poolers:

            # create an sdr that represents the activated neurons in each regional spatial pooler
            #
            spatial_neurons_sdr = SDR()
            for region in pors:

                # get the neurons activated in this regional spatial pooler
                #
                neuron_activations = {pors[region]['activations'][n_idx]['neuron_key']: pors[region]['activations'][n_idx]['similarity']
                                      for n_idx in range(len(pors[region]['activations']))}

                # add this region's encoding
                #
                spatial_neurons_sdr.add_encoding(enc_key=(region,), encoding=neuron_activations)

            # learn the association between each region
            #
            pors['_association'] = self.poolers['_association'].learn_pattern(sdr=spatial_neurons_sdr)

        # train the temporal pooler if required
        # note to train temporal pooler also need an association pooler to have been trained
        #
        if '_temporal' in self.poolers and '_association' in self.poolers:

            # create an sdr that represents the neurons activated in the association pooler
            #
            association_sdr = SDR()

            # get the association's activated neurons
            #
            neuron_activations = {pors['_association']['activations'][n_idx]['neuron_key']: pors['_association']['activations'][n_idx]['similarity']
                                  for n_idx in range(len(pors['_association']['activations']))}

            # add to the encoding - note using temporal key = 0 to represent the current association value
            #
            association_sdr.add_encoding(enc_key=('_association',), encoding=neuron_activations, temporal_key=0)

            # add in the context from the long short term memory
            #
            association_sdr.copy_from(sdr=self.lstm_sdr, from_temporal_key=0, to_temporal_key=1)

            # learn the active neurons from the association of activated neurons
            #
            pors['_temporal'] = self.poolers['_temporal'].learn_pattern(sdr=association_sdr)

            # update the long short term memory that will be the context for the next training session
            #
            lstm_sdr = SDR()
            lstm_sdr.add_encoding(enc_key=('_association',), encoding=neuron_activations)
            self.lstm_sdr.learn(sdr=lstm_sdr, learn_rate=self.temporal_params['lstm_learn_rate'])

        return pors

    def query(self, sdrs: Union[list, dict], similarity_threshold: float = None, decode: bool = True) -> dict:
        pors = {}

        # if we are given a list assume we will need to query the regional spatial poolers, then the temporal pooler and then the temporal pooler
        #
        if isinstance(sdrs, list) and '_association' in self.poolers and '_temporal' in self.poolers:

            lstm_sdr = SDR()
            association_sdr = SDR()
            for item in sdrs:

                # build up the association sdr to query from the activated spatial neurons for each region
                #
                spatial_sdr = SDR()

                for region in item:
                    pors[region] = self.poolers[region].query(sdr=item[region],
                                                              similarity_threshold=similarity_threshold,
                                                              decode=decode)

                    spatial_neuron_activations = {pors[region]['activations'][n_idx]['neuron_key']: pors[region]['activations'][n_idx]['similarity']
                                                  for n_idx in range(len(pors[region]['activations']))}
                    spatial_sdr.add_encoding(enc_key=(region,), encoding=spatial_neuron_activations)

                # query the association pooler
                #
                pors['_association'] = self.poolers['_association'].query(sdr=spatial_sdr,
                                                                          similarity_threshold=similarity_threshold,
                                                                          decode=decode)

                # build up the lstm SDR with the association neuron activations
                #
                association_neuron_activations = {pors['_association']['activations'][n_idx]['neuron_key']: pors['_association']['activations'][n_idx]['similarity']
                                                  for n_idx in range(len(pors['_association']['activations']))}

                # note the association temporal key of 1 ensures this is the context
                #
                association_sdr.add_encoding(enc_key=('_association',), encoding=association_neuron_activations, temporal_key=1)

                lstm_sdr.learn(sdr=association_sdr, learn_rate=self.temporal_params['lstm_learn_rate'])

            # query the temporal pooler
            #
            pors['_temporal'] = self.poolers['_temporal'].query(sdr=lstm_sdr,
                                                                similarity_threshold=similarity_threshold,
                                                                decode=decode)

            # now get the association neurons for each region from the bmu temporal pooler neuron
            # and retrieve any missing region spatial neurons
            #
            for sdr_key in pors['_temporal']['sdr']:

                if sdr_key[TEMPORAL_IDX] == 0:

                    # region is the first item in the sdr_ley enc_type tuple
                    #
                    region = sdr_key[ENC_IDX][0]

                    # if this region is not already in pors then fill in the gaps
                    #
                    if region not in pors:
                        pors[region] = {'activations': [{'neuron_key': neuron_key, 'similarity': pors['_temporal']['sdr'][sdr_key][neuron_key]}
                                                        for neuron_key in pors['_temporal']['sdr'][sdr_key]]}

                        pors[region]['activations'].sort(key=lambda x: x['similarity'], reverse=True)
                        pors[region]['bmu_key'] = pors[region]['activations'][0]['neuron_key']
                        pors[region]['sdr'] = self.poolers[region].get_neuron(neuron_key=pors[region]['bmu_key'], decode=decode)['pattern_sdr']

        # else not provided a dict with regional sdrs to query
        #
        elif isinstance(sdrs, dict):

            # query each regional spatial pooler
            #
            for region in sdrs:
                pors[region] = self.poolers[region].query(sdr=sdrs[region],
                                                          similarity_threshold=similarity_threshold,
                                                          decode=decode)
            # if we trained an association pooler we will also be able to retrieve the other associated regional spatial pooler neurons
            #
            if '_association' in self.poolers:
                # build up the association sdr to query
                #
                spatial_sdr = SDR()
                for region in pors:

                    neuron_activations = {pors[region]['activations'][n_idx]['neuron_key']: pors[region]['activations'][n_idx]['similarity']
                                          for n_idx in range(len(pors[region]['activations']))}
                    spatial_sdr.add_encoding(enc_key=(region,), encoding=neuron_activations)

                # query the association pooler
                #
                pors['_association'] = self.poolers['_association'].query(sdr=spatial_sdr,
                                                                          similarity_threshold=similarity_threshold,
                                                                          decode=decode)

                # now get the spatial neurons for each region from the bmu association pooler neuron
                # and retrieve any missing region spatial neurons
                #
                for sdr_key in pors['_association']['sdr']:

                    # region is the first item in the sdr_ley enc_type tuple
                    #
                    region = sdr_key[ENC_IDX][0]

                    # if this region is not already in pors then fill in the gaps
                    #
                    if region not in pors:
                        pors[region] = {'activations': [{'neuron_key': neuron_key, 'similarity': pors['_association']['sdr'][sdr_key][neuron_key]}
                                                        for neuron_key in pors['_association']['sdr'][sdr_key]]}

                        pors[region]['activations'].sort(key=lambda x: x['similarity'], reverse=True)
                        pors[region]['bmu_key'] = pors[region]['activations'][0]['neuron_key']
                        pors[region]['sdr'] = self.poolers[region].get_neuron(neuron_key=pors[region]['bmu_key'], decode=decode)['pattern_sdr']

        return pors

