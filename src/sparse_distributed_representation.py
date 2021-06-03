#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy, copy
import cython


TEMPORAL_IDX: cython.int = 0
ENC_IDX: cython.int = 1


@cython.cclass
class SDR(object):

    # declare these class instance attributes with cython
    #
    encoding = cython.declare(dict, visibility='public')
    encoders = cython.declare(dict, visibility='public')

    def __init__(self, sdr=None):

        # a map of keys to bit encodings
        #
        self.encoding = {}

        # the encoders used
        #
        self.encoders = {}

        if sdr is not None:
            self.copy(sdr)

    def to_dict(self, decode: bool = True, max_bit_weight: float = 1.0):
        dict_sdr: dict = {}
        sdr_key: tuple

        if decode:
            dict_sdr['encoding'] = self.decode(max_bit_weight)
        else:
            dict_sdr['encoding'] = deepcopy(self.encoding)

        dict_sdr['encoders'] = {sdr_key: self.encoders[sdr_key].type if self.encoders[sdr_key] is not None else None
                                for sdr_key in self.encoders}

        return dict_sdr

    def copy(self, sdr):
        self.encoding = deepcopy(sdr.encoding)
        self.encoders = copy(sdr.encoders)

    def copy_from(self, sdr, from_temporal_key: cython.int = 0, to_temporal_key: cython.int = 0):
        sdr_key: tuple
        for sdr_key in sdr.encoding:
            if sdr_key[TEMPORAL_IDX] == from_temporal_key:
                self.encoding[(to_temporal_key, sdr_key[ENC_IDX])] = sdr.encoding[sdr_key]
                if sdr_key[ENC_IDX] not in self.encoders:
                    self.encoders[sdr_key[ENC_IDX]] = sdr.encoders[sdr_key[ENC_IDX]]

    def add_encoding(self, enc_key: tuple, temporal_key: cython.int = 0, value=None, encoding: dict = None, encoder=None):

        bit: cython.int

        # the encoding key consists of a tuple of temporal key, encoding key
        #
        sdr_key: tuple = (temporal_key, enc_key)

        if encoding is None:
            self.encoding[sdr_key] = {bit: 1.0 for bit in encoder.encode(value)}
            self.encoders[enc_key] = encoder
        else:
            self.encoding[sdr_key] = encoding
            self.encoders[enc_key] = None

    def decode(self, max_bit_weight: float = 1.0) -> dict:
        decode_sdr: dict = {}
        key: tuple

        for key in self.encoding:
            if self.encoders[key[ENC_IDX]] is not None:
                decode_sdr[key] = self.encoders[key[ENC_IDX]].decode(self.encoding[key], max_bit_weight)
            else:
                # convert from frequency to probability
                #
                decode_sdr[key] = [(bit, self.encoding[key][bit] / max_bit_weight) for bit in self.encoding[key]]
                decode_sdr[key].sort(key=lambda x: x[1], reverse=True)
        return decode_sdr

    def learn_delta(self, sdr, learn_temporal_keys: set = None, learn_enc_keys: set = None, learn_rate: float = 1.0, prune_threshold: float = 0.01):

        temporal_keys: set
        temporal_key: cython.int
        bit: cython.int
        sdr_key: tuple

        keys_to_process = set(self.encoding.keys()) | set(sdr.encoding.keys())

        if len(self.encoding) == 0:
            learn_rate = 1.0

        for sdr_key in keys_to_process:
            if ((learn_temporal_keys is None or sdr_key[TEMPORAL_IDX] in learn_temporal_keys) and
                    (learn_enc_keys is None or sdr_key[ENC_IDX] in learn_enc_keys)):

                # if sdr_key not in self then learn bits and encoder
                #
                if sdr_key not in self.encoding:
                    self.encoding[sdr_key] = {bit: sdr.encoding[sdr_key][bit] * learn_rate
                                              for bit in sdr.encoding[sdr_key]
                                              if sdr.encoding[sdr_key][bit] * learn_rate > prune_threshold}
                    self.encoders[sdr_key[ENC_IDX]] = sdr.encoders[sdr_key[ENC_IDX]]

                # if sdr_key in both self and sdr then process each bit
                #
                elif sdr_key in self.encoding and sdr_key in sdr.encoding:

                    # will need to process all bits
                    #
                    bits_to_process = set(self.encoding[sdr_key].keys()) | set(sdr.encoding[sdr_key].keys())

                    for bit in bits_to_process:

                        # if we don't have bit learn it if its above the adjusted prune threshold
                        #
                        if bit not in self.encoding[sdr_key]:
                            bit_value = sdr.encoding[sdr_key][bit] * learn_rate
                            if bit_value > prune_threshold:
                                self.encoding[sdr_key][bit] = bit_value

                        # if bit is in both the calc bit value and assign it if above prune_threshold else delete it
                        #
                        elif bit in self.encoding[sdr_key] and bit in sdr.encoding[sdr_key]:
                            bit_value = self.encoding[sdr_key][bit] + ((sdr.encoding[sdr_key][bit] - self.encoding[sdr_key][bit]) * learn_rate)
                            if bit_value > prune_threshold:
                                self.encoding[sdr_key][bit] = bit_value
                            else:
                                del self.encoding[sdr_key][bit]

                        # if bit only in this sdr then decay bit value and delete it of not above prune_threshold
                        #
                        elif bit in self.encoding[sdr_key]:
                            bit_value = self.encoding[sdr_key][bit] - (self.encoding[sdr_key][bit] * learn_rate)
                            if bit_value > prune_threshold:
                                self.encoding[sdr_key][bit] = bit_value
                            else:
                                del self.encoding[sdr_key][bit]

                # sdr_key only in self so decay bit values, deleting if below prune threshold
                #
                else:
                    bits_to_process = list(self.encoding[sdr_key].keys())
                    for bit in bits_to_process:
                        bit_value = self.encoding[sdr_key][bit] - (self.encoding[sdr_key][bit] * learn_rate)
                        if bit_value > prune_threshold:
                            self.encoding[sdr_key][bit] = bit_value
                        else:
                            del self.encoding[sdr_key][bit]

    def learn_frequency(self, sdr, learn_temporal_keys: set = None, learn_enc_keys: set = None, min_frequency_prune: int = None):

        temporal_keys: set
        temporal_key: cython.int
        bit: cython.int
        sdr_key: tuple

        for sdr_key in sdr.encoding.keys():
            if ((learn_temporal_keys is None or sdr_key[TEMPORAL_IDX] in learn_temporal_keys) and
                    (learn_enc_keys is None or sdr_key[ENC_IDX] in learn_enc_keys)):

                # if sdr_key not in self then learn bits and encoder
                #
                if sdr_key not in self.encoding:
                    self.encoding[sdr_key] = {bit: sdr.encoding[sdr_key][bit]
                                              for bit in sdr.encoding[sdr_key]
                                              if sdr.encoding[sdr_key][bit]}
                    self.encoders[sdr_key[ENC_IDX]] = sdr.encoders[sdr_key[ENC_IDX]]

                # if sdr_key in both self and sdr then process each bit
                #
                else:
                    # will need to process all bits
                    #
                    for bit in sdr.encoding[sdr_key].keys():

                        # if we don't have bit learn it if its above the adjusted prune threshold
                        #
                        if bit not in self.encoding[sdr_key]:
                            self.encoding[sdr_key][bit] = sdr.encoding[sdr_key][bit]
                        else:
                            self.encoding[sdr_key][bit] += sdr.encoding[sdr_key][bit]

        if min_frequency_prune is not None:
            self.encoding = {sdr_key: {bit: self.encoding[sdr_key][bit]
                                       for bit in self.encoding[sdr_key]
                                       if self.encoding[sdr_key][bit] >= min_frequency_prune}
                             for sdr_key in self.encoding.keys()}
