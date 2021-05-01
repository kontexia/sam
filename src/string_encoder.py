#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from src.encoder import Encoder


class StringEncoder(Encoder):
    def __init__(self, name: str = 'sting', n_bits: int = 40, enc_size: int = 2048, bit_offset: int = 0, seed=12345):
        Encoder.__init__(self, encoder_type='string', name=name, n_bits=n_bits, enc_size=enc_size, bit_offset=bit_offset, seed=seed)

    def encode(self, string):

        if string in self.encodings:
            enc = self.encodings[string]
        else:

            # set the state of the random generator
            #
            random.setstate(Encoder.rand_states[self.name])

            enc = set(random.sample(population=self.bit_population, k=self.n_bits))

            Encoder.rand_states[self.name] = random.getstate()

            self.encodings[string] = enc

            # maintain the mapping of bits to bucket
            #
            for bit in enc:
                if bit not in self.bits:
                    self.bits[bit] = {string}
                else:
                    self.bits[bit].add(string)

        return enc

    def decode(self, enc) -> list:
        strings = {}

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

        # sum the weights for the strings associated with the bits in the encoding
        #
        total_weight = 0.0
        for bit in enc:
            for string in self.bits[bit]:
                if string not in strings:
                    strings[string] = enc[bit]
                else:
                    strings[string] += enc[bit]
                total_weight += enc[bit]

        strings = [(string, strings[string] / total_weight) for string in strings]
        strings.sort(key=lambda x: x[1], reverse=True)

        return strings
