#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy, copy
import cython


@cython.cclass
class SDR_v1(object):

    # declare these class instance attributes with cython
    #
    encoding = cython.declare(dict, visibility='public')
    encoders = cython.declare(dict, visibility='public')

    def __init__(self, sdr=None):
        self.encoding = {}
        self.encoders = {}

        if sdr is not None:
            self.copy(sdr)

    def to_dict(self, decode: bool = True):
        d_sdr: dict = {}
        enc_key: str

        if decode:
            d_sdr['encoding'] = self.decode()
        else:
            d_sdr['encoding'] = deepcopy(self.encoding)

        d_sdr['encoders'] = {enc_key: self.encoders[enc_key].type if self.encoders[enc_key] is not None else None
                             for enc_key in self.encoders}

        return d_sdr

    def copy(self, sdr):
        temporal_key: int
        enc_key: str
        bit: cython.int

        self.encoding = {temporal_key: {enc_key: {bit: sdr.encoding[temporal_key][enc_key][bit]
                                                  for bit in sdr.encoding[temporal_key][enc_key]}
                                        for enc_key in sdr.encoding[temporal_key]}
                         for temporal_key in sdr.encoding}

        self.encoders = {enc_key: sdr.encoders[enc_key] for enc_key in sdr.encoders}

    def copy_from(self, sdr, from_temporal_key: cython.int = 0, to_temporal_key: cython.int = 0):
        enc_key: str
        bit: cython.int

        if from_temporal_key in sdr.encoding:
            self.encoding[to_temporal_key] = {enc_key: {bit: sdr.encoding[from_temporal_key][enc_key][bit]
                                                        for bit in sdr.encoding[from_temporal_key][enc_key]}
                                              for enc_key in sdr.encoding[from_temporal_key]}

        for enc_key in sdr.encoders:
            if enc_key not in self.encoders:
                self.encoders[enc_key] = sdr.encoders[enc_key]

    def add_encoding(self, enc_key: str, encoder=None, value=None, encoding: dict = None, temporal_key: cython.int = 0):

        enc_key: str
        bit: cython.int

        if temporal_key not in self.encoding:
            if encoding is None:
                self.encoding[temporal_key] = {enc_key: {bit: 1.0 for bit in encoder.encode(value)}}
            else:
                self.encoding[temporal_key] = {enc_key: encoding}

        else:
            if encoding is None:
                self.encoding[temporal_key][enc_key] = {bit: 1.0 for bit in encoder.encode(value)}
            else:
                self.encoding[temporal_key] = {enc_key: encoding}

        if encoding is None:
            self.encoders[enc_key] = encoder
        else:
            self.encoders[enc_key] = None

    def decode(self) -> dict:
        temporal_key: cython.int
        enc_key: str
        dec_sdr: dict

        dec_sdr = {temporal_key: {enc_key: self.encoders[enc_key].decode(self.encoding[temporal_key][enc_key]) if self.encoders[enc_key] is not None else self.encoding[temporal_key][enc_key]
                                  for enc_key in self.encoding[temporal_key]}
                   for temporal_key in self.encoding}
        return dec_sdr

    def learn(self, sdr, learn_temporal_keys: set = None, learn_enc_keys: set = None, learn_rate: float = 1.0, prune_threshold: float = 0.01):
        temporal_keys: set
        temporal_key: cython.int
        enc_key: str
        bit: cython.int

        # get the union of temporal keys to deal with new or existing temporal keys
        #
        temporal_keys = set(self.encoding.keys()) | set(sdr.encoding.keys())
        for temporal_key in temporal_keys:

            # only learn required temporal keys
            #
            if learn_temporal_keys is None or temporal_key in learn_temporal_keys:

                # if this temporal key not in this sdr then copy over incoming data
                #
                if temporal_key not in self.encoding:
                    self.encoding[temporal_key] = {enc_key: {bit: sdr.encoding[temporal_key][enc_key][bit] * learn_rate
                                                             for bit in sdr.encoding[temporal_key][enc_key]
                                                             if sdr.encoding[temporal_key][enc_key][bit] * learn_rate > prune_threshold}
                                                   for enc_key in sdr.encoding[temporal_key]
                                                   if learn_enc_keys is None or enc_key in learn_enc_keys}
                else:

                    # get the union of enc_keys in this temporal key to deal with new or existing enc_keys
                    #
                    enc_keys = set(self.encoding[temporal_key]) | set(sdr.encoding[temporal_key])
                    for enc_key in enc_keys:

                        # make sure we have an encoder
                        #
                        if enc_key in sdr.encoders:
                            self.encoders[enc_key] = sdr.encoders[enc_key]

                        # only learn required enc_keys
                        #
                        if learn_enc_keys is None or enc_key in learn_enc_keys:

                            # if the enc_key is not in this sdr then copy over incoming data
                            #
                            if enc_key not in self.encoding[temporal_key]:
                                self.encoding[temporal_key][enc_key] = {bit: sdr.encoding[temporal_key][enc_key][bit] * learn_rate
                                                                        for bit in sdr.encoding[temporal_key][enc_key]
                                                                        if sdr.encoding[temporal_key][enc_key][bit] * learn_rate > prune_threshold}
                            else:
                                # get union of bits to deal with existing and new bits
                                #
                                bits = set(self.encoding[temporal_key][enc_key]) | set(sdr.encoding[temporal_key][enc_key])
                                for bit in bits:

                                    # copy over bit weight if new
                                    #
                                    if bit not in self.encoding[temporal_key][enc_key]:
                                        self.encoding[temporal_key][enc_key][bit] = sdr.encoding[temporal_key][enc_key][bit] * learn_rate

                                    # else if bit not in incoming data the decay existing bit weight
                                    #
                                    elif bit not in sdr.encoding[temporal_key][enc_key]:
                                        self.encoding[temporal_key][enc_key][bit] -= self.encoding[temporal_key][enc_key][bit] * learn_rate

                                    # else learn new bit weight
                                    #
                                    else:
                                        self.encoding[temporal_key][enc_key][bit] += (sdr.encoding[temporal_key][enc_key][bit] - self.encoding[temporal_key][enc_key][bit]) * learn_rate

                                    # delete the bit if falls below threshold
                                    #
                                    if self.encoding[temporal_key][enc_key][bit] <= prune_threshold:
                                        del self.encoding[temporal_key][enc_key][bit]



@cython.cclass
class SDR_v2(object):

    # declare these class instance attributes with cython
    #
    encoding = cython.declare(dict, visibility='public')
    encoders = cython.declare(dict, visibility='public')

    def __init__(self, sdr=None):
        self.encoding = {}
        self.encoders = {}

        if sdr is not None:
            self.copy(sdr)

    def to_dict(self, decode: bool = True):
        dict_sdr: dict = {}
        key: tuple

        if decode:
            dict_sdr['encoding'] = self.decode()
        else:
            dict_sdr['encoding'] = deepcopy(self.encoding)

        dict_sdr['encoders'] = {key: self.encoders[key].type if self.encoders[key] is not None else None
                                for key in self.encoders}

        return dict_sdr

    def copy(self, sdr):
        self.encoding = deepcopy(sdr.encoding)
        self.encoders = copy(sdr.encoders)

    def copy_from(self, sdr, from_temporal_key: cython.int = 0, to_temporal_key: cython.int = 0):
        key: tuple
        for key in sdr.encoding:
            if key[1] == from_temporal_key:
                self.encoding[(key[0], to_temporal_key, key[2])] = sdr.encoding[key]
                if (key[0], key[2]) not in self.encoders:
                    self.encoders[(key[0], key[2])] = sdr.encoders[(key[0], key[2])]

    def add_encoding(self, enc_key: tuple, temporal_key: cython.int = 0, group: str = None, value=None, encoding: dict = None, encoder=None):

        bit: cython.int

        # the encoding key consists of a tuple of group, temporal key, encoding key
        #
        key = (group, temporal_key, enc_key)

        if encoding is None:
            self.encoding[key] = {bit: 1.0 for bit in encoder.encode(value)}
            self.encoders[(group, enc_key)] = encoder
        else:
            self.encoding[key] = encoding
            self.encoders[(group, enc_key)] = None

    def decode(self) -> dict:
        decode_sdr: dict = {}
        key: tuple

        for key in self.encoding:
            if self.encoders[(key[0], key[2])] is not None:
                decode_sdr[key] = self.encoders[(key[0], key[2])].decode(self.encoding[key])
            else:
                decode_sdr[key] = self.encoding[key]
        return decode_sdr

    def learn(self, sdr, learn_group_keys: set = None, learn_temporal_keys: set = None, learn_enc_keys: set = None, learn_rate: float = 1.0, prune_threshold: float = 0.01):

        temporal_keys: set
        temporal_key: cython.int
        enc_key: str
        bit: cython.int

        keys_to_process = set(self.encoding.keys()) | set(sdr.encoding.keys())

        for key in keys_to_process:
            if ((learn_group_keys is None or key[0] in learn_group_keys) and
                    (learn_temporal_keys is None or key[1] in learn_temporal_keys) and
                    (learn_enc_keys is None or key[2] in learn_enc_keys)):

                # if key not in self then learn bits and encoder
                #
                if key not in self.encoding:
                    self.encoding[key] = {bit: sdr.encoding[key][bit] * learn_rate
                                          for bit in sdr.encoding[key]
                                          if sdr.encoding[key][bit] * learn_rate > prune_threshold}
                    self.encoders[(key[0], key[2])] = sdr.encoders[(key[0], key[2])]

                # if key in both self and sdr then process each bit
                #
                elif key in self.encoding and key in sdr.encoding:

                    # will need to process all bits
                    #
                    bits_to_process = set(self.encoding[key].keys()) | set(sdr.encoding[key].keys())

                    for bit in bits_to_process:

                        # if we don't have bit learn it if its above the adjusted prune threshold
                        #
                        if bit not in self.encoding[key]:
                            bit_value = sdr.encoding[key][bit] * learn_rate
                            if bit_value > prune_threshold:
                                self.encoding[key][bit] = bit_value

                        # if bit is in both the calc bit value and assign it if above prune_threshold else delete it
                        #
                        elif bit in self.encoding[key] and bit in sdr.encoding[key]:
                            bit_value = self.encoding[key][bit] + ((sdr.encoding[key][bit] - self.encoding[key][bit]) * learn_rate)
                            if bit_value > prune_threshold:
                                self.encoding[key][bit] = bit_value
                            else:
                                del self.encoding[key][bit]

                        # if bit only in this sdr then decay bit value and delete it of not above prune_threshold
                        #
                        elif bit in self.encoding[key]:
                            bit_value = self.encoding[key][bit] - (self.encoding[key][bit] * learn_rate)
                            if bit_value > prune_threshold:
                                self.encoding[key][bit] = bit_value
                            else:
                                del self.encoding[key][bit]

                # key only in self so decay bit values, deleting if below prune threshold
                #
                else:
                    for bit in self.encoding[key]:
                        bit_value = self.encoding[key][bit] - (self.encoding[key][bit] * learn_rate)
                        if bit_value > prune_threshold:
                            self.encoding[key][bit] = bit_value
                        else:
                            del self.encoding[key][bit]


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

    def to_dict(self, decode: bool = True):
        dict_sdr: dict = {}
        sdr_key: tuple

        if decode:
            dict_sdr['encoding'] = self.decode()
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

    def decode(self) -> dict:
        decode_sdr: dict = {}
        key: tuple

        for key in self.encoding:
            if self.encoders[key[ENC_IDX]] is not None:
                decode_sdr[key] = self.encoders[key[ENC_IDX]].decode(self.encoding[key])
            else:
                decode_sdr[key] = self.encoding[key]
        return decode_sdr

    def learn(self, sdr, learn_temporal_keys: set = None, learn_enc_keys: set = None, learn_rate: float = 1.0, prune_threshold: float = 0.01):

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
                    for bit in self.encoding[sdr_key]:
                        bit_value = self.encoding[sdr_key][bit] - (self.encoding[sdr_key][bit] * learn_rate)
                        if bit_value > prune_threshold:
                            self.encoding[sdr_key][bit] = bit_value
                        else:
                            del self.encoding[sdr_key][bit]
