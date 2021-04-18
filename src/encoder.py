#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random


class Encoder(object):

    rand_states = {}

    def __init__(self, encoder_type, n_bits: int = 40, enc_size: int = 2048, seed=None):
        self.type = encoder_type
        self.encodings = {}
        self.bits = {}
        self.n_bits = n_bits
        self.enc_size = enc_size
        if seed is not None:
            random.seed(seed)

        if encoder_type not in Encoder.rand_states:
            Encoder.rand_states[encoder_type] = random.getstate()
