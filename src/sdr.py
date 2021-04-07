#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from copy import deepcopy


class SDR(object):

    def __init__(self, sdr=None, enc_type=None, value=None, enc_properties=None, encoder=None, bit_weight=1.0):

        self.encodings = {}
        self.properties = {}

        # copy an sdr if provided
        # note don't copy encoder just take reference
        #
        if sdr is not None:
            self.copy(sdr=sdr)

        # else process a new value
        #
        elif enc_type is not None and value is not None and encoder is not None:
            self.add_encoding(enc_type=enc_type, value=value, encoder=encoder, enc_properties=enc_properties, bit_weight=bit_weight)

    def get_max_distance(self, search_types=None):

        dist = sum([self.properties[enc_type]['encoder'].n_bits
                    for enc_type in self.properties
                    if search_types is None or enc_type in search_types])

        return dist

    def remove_prefix(self, enc_type_prefix):
        self.encodings = {enc_type.replace(enc_type_prefix, ''): self.encodings[enc_type] for enc_type in self.encodings}
        self.properties = {enc_type.replace(enc_type_prefix, ''): self.properties[enc_type] for enc_type in self.properties}

    def copy(self, sdr, enc_type_prefix=None):
        if enc_type_prefix is not None:
            self.encodings = {f'{enc_type_prefix}_{enc_type}': deepcopy(sdr.encodings[enc_type]) for enc_type in sdr.encodings}
            self.properties = {f'{enc_type_prefix}_{enc_type}': {prop: sdr.properties[enc_type][prop] for prop in sdr.properties[enc_type]} for enc_type in sdr.properties}
        else:
            self.encodings = deepcopy(sdr.encodings)
            self.properties = {enc_type: {prop: sdr.properties[enc_type][prop] for prop in sdr.properties[enc_type]} for enc_type in sdr.properties}

    def add_encoding(self, enc_type, value, encoder, enc_properties=None, bit_weight=1.0):
        if enc_properties is not None:
            self.properties[enc_type] = {prop: enc_properties[prop] for prop in enc_properties}

        self.properties[enc_type] = {'encoder': encoder}

        enc = encoder.encode(value)
        self.encodings[enc_type] = {e: bit_weight for e in enc}

    def get_enc_types(self):
        return set(self.encodings.keys())

    def get_enc_properties(self, enc_type):
        if enc_type in self.properties:
            return self.properties[enc_type]
        else:
            return {}

    def decode(self):
        dec = {enc_type: self.properties[enc_type]['encoder'].decode(self.encodings[enc_type]) for enc_type in self.encodings}
        return dec

    def to_dict(self, decode=False):
        if decode:
            d_sdr = {'encodings': self.decode()}
        else:
            d_sdr = {'encodings': deepcopy(self.encodings)}
        d_sdr['properties'] = {enc_type: {prop: self.properties[enc_type][prop] for prop in self.properties[enc_type]} for enc_type in self.properties}
        return d_sdr

    def overlap(self, sdr, search_types=None):

        # get the filtered intersection of enc_types to compare
        #
        if search_types is not None:
            enc_types = ({enc_type
                          for enc_type in self.encodings.keys()
                          if enc_type in search_types} &
                         {enc_type
                          for enc_type in sdr.encodings.keys()
                          if enc_type in search_types})
        else:
            enc_types = set(self.encodings.keys()) & set(sdr.encodings.keys())

        por = {'overlap': 0, 'norm_overlap': 0.0}

        por['enc_types'] = {enc_type: sum([min(self.encodings[enc_type][b], sdr.encodings[enc_type][b])
                                           for b in set(self.encodings[enc_type].keys()) & set(sdr.encodings[enc_type].keys())])
                            for enc_type in enc_types}

        por['overlap'] = sum([por['enc_types'][enc_type] for enc_type in por['enc_types']])

        if len(enc_types) > 0:
            por['norm_overlap'] = por['overlap'] / len(enc_types)

        return por

    def distance(self, sdr, search_types=None):
        por = self.overlap(sdr=sdr, search_types=search_types)
        por['max_distance'] = max(self.get_max_distance(search_types=search_types), sdr.get_max_distance(search_types=search_types))
        por['distance'] = por['max_distance'] - por['overlap']
        if por['max_distance'] > 0.0:
            por['similarity'] = por['overlap'] / por['max_distance']
        else:
            por['similarity'] = 0.0

        return por

    def learn(self, sdr, learn_rate=1.0, learn_types=None, prune=0.01):

        # if this sdr is empty then learn at fastest rate
        #
        if len(self.encodings) == 0:
            learn_rate = 1.0

        if learn_types is not None:
            enc_types = ({enc_type
                         for enc_type in self.encodings.keys()
                         if enc_type in learn_types} |
                         {enc_type
                          for enc_type in sdr.encodings.keys()
                          if enc_type in learn_types})

        else:
            enc_types = set(self.encodings.keys()) | set(sdr.encodings.keys())

        for enc_type in enc_types:
            if enc_type in self.encodings and enc_type in sdr.encodings:
                self_bits = set(self.encodings[enc_type].keys())
                sdr_bits = set(sdr.encodings[enc_type].keys())
                bits = self_bits | sdr_bits
                for bit in bits:
                    if bit in self.encodings[enc_type] and bit in sdr.encodings[enc_type]:
                        self.encodings[enc_type][bit] += (sdr.encodings[enc_type][bit] - self.encodings[enc_type][bit]) * learn_rate
                    elif bit in self.encodings[enc_type]:
                        self.encodings[enc_type][bit] -= self.encodings[enc_type][bit] * learn_rate
                    else:
                        self.encodings[enc_type][bit] = sdr.encodings[enc_type][bit] * learn_rate

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]

            elif enc_type in self.encodings:
                for bit in list(self.encodings[enc_type]):
                    self.encodings[enc_type][bit] -= (self.encodings[enc_type][bit] * learn_rate)

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]
            else:
                self.encodings[enc_type] = {bit: sdr.encodings[enc_type][bit] * learn_rate for bit in sdr.encodings[enc_type]}

                # copy over the properties
                #
                self.properties[enc_type] = {prop: sdr.properties[enc_type][prop] for prop in sdr.properties[enc_type]}

            # prune the bit type if empty
            #
            if len(self.encodings[enc_type]) == 0:
                del self.encodings[enc_type]
                del self.properties[enc_type]

    def merge(self, sdr, weight=1.0, enc_type_prefix=None):
        for enc_type in sdr.encodings:

            if enc_type_prefix is not None:
                self_enc_type = f'{enc_type_prefix}_{enc_type}'
            else:
                self_enc_type = enc_type

            if self_enc_type in self.encodings:
                for bit in sdr.encodings[enc_type].keys():
                    if bit in self.encodings[self_enc_type]:
                        self.encodings[self_enc_type][bit] += sdr.encodings[enc_type][bit] * weight
                    else:
                        self.encodings[self_enc_type][bit] = sdr.encodings[enc_type][bit] * weight
            else:
                self.encodings[self_enc_type] = {bit: sdr.encodings[enc_type][bit] * weight for bit in sdr.encodings[enc_type]}

                # copy over the properties
                #
                self.properties[self_enc_type] = {prop: sdr.properties[enc_type][prop] for prop in sdr.properties[enc_type]}


if __name__ == '__main__':
    from src.numeric_encoder import NumericEncoder
    import time

    encoder = NumericEncoder(min_step=1,
                             n_bits=40,
                             enc_size=2048,
                             seed=12345)

    sdr_1 = SDR(enc_type='volume', value=100, encoder=encoder)

    sdr_2 = SDR(enc_type='volume', value=210, encoder=encoder)

    d = sdr_1.distance(sdr_2)

    o = sdr_1.overlap(sdr_2)

    m_dist = sdr_1.get_max_distance()

    sdr_3 = SDR(sdr=sdr_2)

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    d_1 = sdr_1.distance(sdr=sdr_3)

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    d_2 = sdr_1.distance(sdr=sdr_3)

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    d_3 = sdr_1.distance(sdr=sdr_3)

    sdr_4 = SDR()

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    start_time = time.time()
    for i in range(1000):
        d_4 = sdr_1.distance(sdr=sdr_3)
    end_time = time.time()
    print((end_time - start_time)/1000)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    d_5 = sdr_1.distance(sdr=sdr_3)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    d_6 = sdr_1.distance(sdr=sdr_3)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    d_7 = sdr_1.distance(sdr=sdr_3)


    print('finished')
