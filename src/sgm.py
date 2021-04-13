#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from copy import deepcopy
import cython

@cython.cclass
class SGM(object):
    encodings = cython.declare(dict, visibility='public')
    properties = cython.declare(dict, visibility='public')
    max_bits = cython.declare(int, visibility='public')

    def __init__(self, sgm=None, enc_type=None, value=None, enc_properties=None, encoder=None, bit_weight=1.0):

        self.encodings = {}
        self.properties = {}
        self.max_bits = 0

        # copy an sgm if provided
        # note don't copy encoder just take reference
        #
        if sgm is not None:
            self.copy(sgm=sgm)

        # else process a new value
        #
        elif enc_type is not None and value is not None and encoder is not None:
            self.add_encoding(enc_type=enc_type, value=value, encoder=encoder, enc_properties=enc_properties, bit_weight=bit_weight)

    def remove_prefix(self, enc_type_prefix: str):
        enc_type: str

        self.encodings = {enc_type.replace(enc_type_prefix, ''): self.encodings[enc_type] for enc_type in self.encodings}
        self.properties = {enc_type.replace(enc_type_prefix, ''): self.properties[enc_type] for enc_type in self.properties}

    def copy(self, sgm, enc_type_prefix: str = None):
        enc_type: str
        prop: str
        self.max_bits = sgm.max_bits

        if enc_type_prefix is not None:
            self.encodings = {f'{enc_type_prefix}_{enc_type}': deepcopy(sgm.encodings[enc_type]) for enc_type in sgm.encodings}
            self.properties = {f'{enc_type_prefix}_{enc_type}': {prop: sgm.properties[enc_type][prop] for prop in sgm.properties[enc_type]} for enc_type in sgm.properties}
        else:
            self.encodings = deepcopy(sgm.encodings)
            self.properties = {enc_type: {prop: sgm.properties[enc_type][prop] for prop in sgm.properties[enc_type]} for enc_type in sgm.properties}

    def add_encoding(self, enc_type: str, value, encoder, enc_properties: dict = None, bit_weight: float = 1.0):

        bit: int
        prop: str

        if enc_properties is not None:
            self.properties[enc_type] = {prop: enc_properties[prop] for prop in enc_properties}

        self.properties[enc_type] = {'encoder': encoder}

        if encoder.n_bits > self.max_bits:
            self.max_bits = encoder.n_bits

        enc = encoder.encode(value)
        self.encodings[enc_type] = {bit: bit_weight for bit in enc}

    def get_enc_types(self) -> set:
        return set(self.encodings.keys())

    def get_enc_properties(self, enc_type: str) -> dict:
        if enc_type in self.properties:
            return self.properties[enc_type]
        else:
            return {}

    def decode(self) -> dict:
        enc_type: str
        dec = {enc_type: self.properties[enc_type]['encoder'].decode(self.encodings[enc_type]) for enc_type in self.encodings}
        return dec

    def to_dict(self, decode: bool = False) -> dict:
        enc_type: str
        prop: str

        if decode:
            d_sdr = {'encodings': self.decode()}
        else:
            d_sdr = {'encodings': deepcopy(self.encodings)}
        d_sdr['properties'] = {enc_type: {prop: self.properties[enc_type][prop] for prop in self.properties[enc_type]} for enc_type in self.properties}
        return d_sdr

    def similarity(self, sgm, search_types: set = None) -> dict:

        enc_type: str
        por: dict
        bit: int

        # if no search_types provided get union of types from both sgms
        #
        if search_types is None:
            search_types = set(self.encodings.keys()) | set(sgm.encodings.keys())

        por = {'enc_types': {}, 'similarity': 0.0}

        # for each enc_type calculate the overlap in bits - which is the sum of minimum weights of each bit
        #
        for enc_type in search_types:

            if enc_type in self.encodings and enc_type in sgm.encodings:
                por['enc_types'][enc_type] = {'overlap': sum([min(self.encodings[enc_type][bit], sgm.encodings[enc_type][bit])
                                                              for bit in set(self.encodings[enc_type].keys()) & set(sgm.encodings[enc_type].keys())])}
            else:
                # if the enc_type is not in both then overlap must be 0
                #
                por['enc_types'][enc_type] = {'overlap': 0}

            # similarity is the overlap divided by the maximum number of bits
            #
            if self.max_bits > 0:
                por['enc_types'][enc_type]['similarity'] = por['enc_types'][enc_type]['overlap'] / self.max_bits
                por['similarity'] += por['enc_types'][enc_type]['similarity']
            else:
                por['enc_types'][enc_type]['similarity'] = 0.0

        # calc the average similarity across the enc_types
        #
        if len(por['enc_types']) > 0:
            por['similarity'] = por['similarity'] / len(por['enc_types'])

        return por

    def learn(self, sgm, learn_rate: float = 1.0, learn_types: set = None, prune: float = 0.01):

        enc_type: str
        enc_types: set
        por: dict
        bits: set
        bit: int
        prop: str

        # if this sgm is empty then learn at fastest rate
        #
        if len(self.encodings) == 0:
            learn_rate = 1.0

        # get the enc_types to learn
        #
        if learn_types is not None:
            enc_types = ({enc_type
                         for enc_type in self.encodings.keys()
                         if enc_type in learn_types} |
                         {enc_type
                          for enc_type in sgm.encodings.keys()
                          if enc_type in learn_types})

        else:
            enc_types = set(self.encodings.keys()) | set(sgm.encodings.keys())

        for enc_type in enc_types:

            if enc_type in self.encodings and enc_type in sgm.encodings:

                # learn te bit weights - ie self bits will move towards the sgm bit weights. if a bit does exist in the sgm encoding then self bit weight is reduced
                #
                self_bits = set(self.encodings[enc_type].keys())
                sdr_bits = set(sgm.encodings[enc_type].keys())
                bits = self_bits | sdr_bits
                for bit in bits:
                    if bit in self.encodings[enc_type] and bit in sgm.encodings[enc_type]:
                        # move the weighting towards the sgm bit weight
                        #
                        self.encodings[enc_type][bit] += (sgm.encodings[enc_type][bit] - self.encodings[enc_type][bit]) * learn_rate
                    elif bit in self.encodings[enc_type]:
                        # reduce the weighting towards zero
                        #
                        self.encodings[enc_type][bit] -= self.encodings[enc_type][bit] * learn_rate
                    else:
                        # learn the weight for this new bit
                        #
                        self.encodings[enc_type][bit] = sgm.encodings[enc_type][bit] * learn_rate

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]

            elif enc_type in self.encodings:
                # as this enc_type doesn't exist in the sgm then reduce the bit weights towards zero
                # note as we can prune the bit we iterate over a copy list
                #
                for bit in list(self.encodings[enc_type]):
                    self.encodings[enc_type][bit] -= (self.encodings[enc_type][bit] * learn_rate)

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]
            else:
                # learn the weights of the new bits
                #
                self.encodings[enc_type] = {bit: sgm.encodings[enc_type][bit] * learn_rate for bit in sgm.encodings[enc_type]}

                # copy over the properties
                #
                self.properties[enc_type] = {prop: sgm.properties[enc_type][prop] for prop in sgm.properties[enc_type]}

            # prune the enc type if it has no bits
            #
            if len(self.encodings[enc_type]) == 0:
                del self.encodings[enc_type]
                del self.properties[enc_type]

    def merge(self, sgm, weight: float = 1.0, enc_type_prefix: str = None):
        enc_type: str
        bit: int
        prop: str

        for enc_type in sgm.encodings:

            if enc_type_prefix is not None:
                self_enc_type = f'{enc_type_prefix}_{enc_type}'
            else:
                self_enc_type = enc_type

            if self_enc_type in self.encodings:
                for bit in sgm.encodings[enc_type].keys():
                    if bit in self.encodings[self_enc_type]:
                        self.encodings[self_enc_type][bit] += sgm.encodings[enc_type][bit] * weight
                    else:
                        self.encodings[self_enc_type][bit] = sgm.encodings[enc_type][bit] * weight
            else:
                self.encodings[self_enc_type] = {bit: sgm.encodings[enc_type][bit] * weight for bit in sgm.encodings[enc_type]}

                # copy over the properties
                #
                self.properties[self_enc_type] = {prop: sgm.properties[enc_type][prop] for prop in sgm.properties[enc_type]}
