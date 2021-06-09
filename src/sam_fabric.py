#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.pooler import Pooler
from src.sparse_distributed_representation import SDR, ENC_IDX, TEMPORAL_IDX
from src.numeric_encoder import NumericEncoder
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
                self.spatial_params[region]['similarity_threshold'] = 0.7
            if 'community_factor' not in self.spatial_params[region]:
                self.spatial_params[region]['community_factor'] = 0.9
            if 'prune_threshold' not in self.spatial_params[region]:
                self.spatial_params[region]['prune_threshold'] = 0.01
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
                self.association_params['similarity_threshold'] = 0.7
            if 'community_factor' not in self.association_params:
                self.association_params['community_factor'] = 0.9
            if 'prune_threshold' not in self.association_params:
                self.association_params['prune_threshold'] = 0.01

            self.poolers['_association'] = Pooler(name='_association',
                                                  similarity_threshold=self.association_params['similarity_threshold'],
                                                  community_factor=self.association_params['community_factor'],
                                                  prune_threshold=self.association_params['prune_threshold'])

        # instantiate the temporal pooler if required
        #
        if self.temporal_params is not None:
            if 'similarity_threshold' not in self.temporal_params:
                self.temporal_params['similarity_threshold'] = 0.7
            if 'community_factor' not in self.temporal_params:
                self.temporal_params['community_factor'] = 0.9
            if 'prune_threshold' not in self.temporal_params:
                self.temporal_params['prune_threshold'] = 0.01
            if 'lstm_len' not in self.temporal_params:
                self.temporal_params['lstm_len'] = 4

            self.poolers['_temporal'] = Pooler(name='_temporal',
                                               similarity_threshold=self.temporal_params['similarity_threshold'],
                                               community_factor=self.temporal_params['community_factor'],
                                               prune_threshold=self.temporal_params['prune_threshold'])
            self.lstm_window = []
        else:
            self.lstm_window = None

        self.similarity_encoder = NumericEncoder(name='Similarity', min_step=0.01)

    def to_dict(self, decode: bool = True) -> dict:

        dict_fabric = {'association_params': self.association_params,
                       'temporal_params': self.temporal_params,
                       'lstm_window': None}

        if self.lstm_window is not None:
            dict_fabric['lstm_window'] = [sdr.to_dict(decode=decode) for sdr in self.lstm_window]

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
                self.poolers[region] = Pooler(name=region,
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
            spatial_sdr = SDR()
            for region in pors:

                # get the neurons activated in this regional spatial pooler
                #
                neuron_activations = {pors[region]['activations'][n_idx]['neuron_key']: 1.0
                                      for n_idx in range(len(pors[region]['activations']))
                                      if pors[region]['activations'][n_idx]['similarity'] >= self.spatial_params[region]['similarity_threshold']}

                # add this region's encoding
                #
                spatial_sdr.add_encoding(enc_key=(region,), encoding=neuron_activations)

            # learn the association between each region
            #
            pors['_association'] = self.poolers['_association'].learn_pattern(sdr=spatial_sdr)

        # train the temporal pooler if required
        # note to train temporal pooler also need an association pooler to have been trained
        #
        if '_temporal' in self.poolers and '_association' in self.poolers:

            # create an sdr that represents the neurons activated in the association pooler
            #
            association_sdr = SDR()

            # get the association's activated neurons
            #
            neuron_activations = {pors['_association']['activations'][n_idx]['neuron_key']: 1.0
                                  for n_idx in range(len(pors['_association']['activations']))
                                  if pors['_association']['activations'][n_idx]['similarity'] >= self.association_params['similarity_threshold']}

            # add to the encoding - note using temporal key = 0 to represent the current association value
            #
            association_sdr.add_encoding(enc_key=('_association',), encoding=neuron_activations, temporal_key=0)

            # only train once the temporal window is populated
            #
            if len(self.lstm_window) == self.temporal_params['lstm_len']:
                context_sdr = SDR(association_sdr)
                # add in the context from the long short term memory
                #
                for t_idx in range(len(self.lstm_window)):
                    context_sdr.copy_from(sdr=self.lstm_window[t_idx], from_temporal_key=0, to_temporal_key=len(self.lstm_window) - t_idx)

                # learn the active neurons from the association of activated neurons
                #
                pors['_temporal'] = self.poolers['_temporal'].learn_pattern(sdr=context_sdr)
            else:
                pors['_temporal'] = {'bmu_similarity': 0.0,
                                     'nos_neurons': 0,
                                     'avg_bmu_similarity': 0.0,
                                     'std_bmu_similarity': 0.0}

            # update the long short term memory that will be the context for the next training session
            #
            self.lstm_window.append(association_sdr)
            if len(self.lstm_window) > self.temporal_params['lstm_len']:
                self.lstm_window.pop(0)

        return pors

    def query(self, sdrs: Union[list, dict], similarity_threshold: float = None, decode: bool = True) -> dict:
        pors = {}

        # if we are given a list assume we will need to query the regional spatial poolers, then the associative pooler and then the temporal pooler
        #
        if isinstance(sdrs, list) and '_association' in self.poolers and '_temporal' in self.poolers:

            temporal_sdr = SDR()

            # assume the list contains sdrs where the first in the list is the oldest and the last in the list is the newest
            # t_idx will be used to refer to the temporal context sdr keys in the temporal pooler
            #
            for t_idx in range(1, min(len(sdrs), self.temporal_params['lstm_len']) + 1):

                # Note use of negative index to get the most recent from the list
                #
                item = sdrs[-t_idx]

                # build up an sdr from the regional spatial neurons to query the association pooler
                #
                association_sdr = SDR()

                for region in item:
                    region_por = self.poolers[region].query(sdr=item[region],
                                                            similarity_threshold=similarity_threshold,
                                                            decode=decode)

                    # get the neurons activated in this regional spatial pooler
                    #
                    neuron_activations = {region_por['activations'][n_idx]['neuron_key']: 1.0
                                          for n_idx in range(len(region_por['activations']))
                                          if region_por['activations'][n_idx]['similarity'] >= self.spatial_params[region]['similarity_threshold']}

                    # add this region's encoding
                    #
                    association_sdr.add_encoding(enc_key=(region,), encoding=neuron_activations)

                # query the association pooler
                #
                assoc_por = self.poolers['_association'].query(sdr=association_sdr,
                                                               similarity_threshold=similarity_threshold,
                                                               decode=decode)

                # update the temporal_sdr query
                #
                neuron_activations = {assoc_por['activations'][n_idx]['neuron_key']: 1.0
                                      for n_idx in range(len(assoc_por['activations']))
                                      if assoc_por['activations'][n_idx]['similarity'] >= self.association_params['similarity_threshold']}

                temporal_sdr.add_encoding(enc_key=('_association',), encoding=neuron_activations, temporal_key=t_idx)

            # query the temporal pooler
            #
            pors['_temporal'] = self.poolers['_temporal'].query(sdr=temporal_sdr,
                                                                similarity_threshold=similarity_threshold,
                                                                decode=decode)

            # now get the association neurons for each region from the bmu temporal pooler neuron
            # and retrieve any missing region spatial neurons
            #
            for temporal_sdr_key in pors['_temporal']['sdr']:

                # only interested in the current association sdr in temporal_key slot 0
                #
                if temporal_sdr_key[TEMPORAL_IDX] == 0 and temporal_sdr_key[ENC_IDX][0] == '_association':

                    # build up the association query results from the temporal query bmu
                    #
                    pors['_association'] = {'activations': []}

                    # the association neurons are

                    for n_idx in range(len(pors['_temporal']['sdr'][temporal_sdr_key])):

                        neuron = pors['_temporal']['sdr'][temporal_sdr_key][n_idx]

                        association_neuron = self.poolers['_association'].get_neuron(neuron_key=neuron[0], decode=decode)
                        pors['_association']['activations'].append({'neuron_key': neuron[0],
                                                                    'similarity': neuron[1],
                                                                    'last_updated': association_neuron['n_bmu'],
                                                                    'sdr': association_neuron['pattern_sdr']['encoding']})

                    pors['_association']['activations'].sort(key=lambda x: x['similarity'], reverse=True)
                    pors['_association']['bmu_key'] = pors['_association']['activations'][0]['neuron_key']
                    pors['_association']['sdr'] = pors['_association']['activations'][0]['sdr']

                    for assoc_sdr_key in pors['_association']['sdr']:

                        # region is the first item in the sdr_ley enc_type tuple
                        #
                        region = assoc_sdr_key[ENC_IDX][0]

                        # if this region is not already in pors then fill in the gaps
                        #
                        if region not in pors:
                            pors[region] = {'activations': []}

                            for n_idx in range(len(pors['_association']['sdr'][assoc_sdr_key])):

                                neuron_key = pors['_association']['sdr'][assoc_sdr_key][n_idx]
                                region_neuron = self.poolers[region].get_neuron(neuron_key=neuron_key[0], decode=decode)
                                pors[region]['activations'].append({'neuron_key': neuron_key[0],
                                                                    'similarity': neuron_key[1],
                                                                    'last_updated': region_neuron['n_bmu'],
                                                                    'sdr': region_neuron['pattern_sdr']['encoding']})

                            pors[region]['activations'].sort(key=lambda x: x['similarity'], reverse=True)
                            pors[region]['bmu_key'] = pors[region]['activations'][0]['neuron_key']
                            pors[region]['sdr'] = pors[region]['activations'][0]['sdr']

                    # we are done so can break out of this loop
                    #
                    break

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
                association_sdr = SDR()
                for region in pors:

                    neuron_activations = {pors[region]['activations'][n_idx]['neuron_key']: 1.0
                                          for n_idx in range(len(pors[region]['activations']))
                                          if pors[region]['activations'][n_idx]['similarity'] >= self.spatial_params[region]['similarity_threshold']}

                    association_sdr.add_encoding(enc_key=(region,), encoding=neuron_activations)

                # query the association pooler
                #
                pors['_association'] = self.poolers['_association'].query(sdr=association_sdr,
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
                        pors[region] = {'activations': []}

                        for n_idx in range(len(pors['_association']['sdr'][sdr_key])):

                            neuron = pors['_association']['sdr'][sdr_key][n_idx]
                            region_neuron = self.poolers[region].get_neuron(neuron_key=neuron[0], decode=decode)
                            pors[region]['activations'].append({'neuron_key': neuron[0],
                                                                'similarity': neuron[1],
                                                                'last_updated': region_neuron['n_bmu'],
                                                                'sdr': region_neuron['pattern_sdr']['encoding']})

                        pors[region]['activations'].sort(key=lambda x: x['similarity'], reverse=True)
                        pors[region]['bmu_key'] = pors[region]['activations'][0]['neuron_key']
                        pors[region]['sdr'] = pors[region]['activations'][0]['sdr']

        return pors
