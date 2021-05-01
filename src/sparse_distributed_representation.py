#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
import cython

@cython.cclass
class SDR(object):

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

        d_sdr['encoders'] = {enc_key: self.encoders[enc_key].type for enc_key in self.encoders}

        return d_sdr

    def copy(self, sdr):
        temporal_key: int
        enc_key: str
        bit: int

        self.encoding = {temporal_key: {enc_key: {bit: sdr.encoding[temporal_key][enc_key][bit]
                                                  for bit in sdr.encoding[temporal_key][enc_key]}
                                        for enc_key in sdr.encoding[temporal_key]}
                         for temporal_key in sdr.encoding}

        self.encoders = {enc_key: sdr.encoders[enc_key] for enc_key in sdr.encoders}

    def copy_from(self, sdr, from_temporal_key: int = 0, to_temporal_key: int = 0):
        enc_key: str
        bit: int

        if from_temporal_key in sdr.encoding:
            self.encoding[to_temporal_key] = {enc_key: {bit: sdr.encoding[from_temporal_key][enc_key][bit]
                                                        for bit in sdr.encoding[from_temporal_key][enc_key]}
                                              for enc_key in sdr.encoding[from_temporal_key]}

        for enc_key in sdr.encoders:
            if enc_key not in self.encoders:
                self.encoders[enc_key] = sdr.encoders[enc_key]

    def add_encoding(self, value, enc_key: str, encoder, temporal_key: int = 0):

        enc_key: str
        bit: int

        if temporal_key not in self.encoding:
            self.encoding[temporal_key] = {enc_key: {bit: 1.0 for bit in encoder.encode(value)}}
        else:
            self.encoding[temporal_key][enc_key] = {bit: 1.0 for bit in encoder.encode(value)}

        self.encoders[enc_key] = encoder

    def decode(self) -> dict:
        temporal_key: int
        enc_key: str
        dec_sdr: dict

        dec_sdr = {temporal_key: {enc_key: self.encoders[enc_key].decode(self.encoding[temporal_key][enc_key])
                                  for enc_key in self.encoding[temporal_key]}
                   for temporal_key in self.encoding}
        return dec_sdr

    def learn(self, sdr, learn_temporal_keys: set = None, learn_enc_keys: set = None, learn_rate: float = 1.0, prune_threshold: float = 0.01):
        temporal_keys: set
        temporal_key: int
        enc_key: str
        bit: int

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
