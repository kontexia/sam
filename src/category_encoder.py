#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.encoder import Encoder
import random


class CategoryEncoder(Encoder):
    def __init__(self, name='category', semantic_categories=None, n_bits: int = 40, enc_size: int = 2048, bit_offset: int = 0, periodic=False, seed=12345):
        Encoder.__init__(self, encoder_type='category', name=name, n_bits=n_bits, enc_size=enc_size, bit_offset=bit_offset, seed=seed)

        self.periodic = periodic
        self.semantic = False

        if semantic_categories is not None:
            self.set_semantic_categories(categories=semantic_categories)

    def set_semantic_categories(self, categories):
        self.semantic = True

        if len(categories) < self.n_bits and self.periodic:
            step = int(self.n_bits / 2)
        else:
            step = 1

        start_bit = 0
        if self.periodic:
            for category_idx in range(len(categories) - 1):
                self.encodings[categories[category_idx]] = []
                for bit in range(start_bit, start_bit + self.n_bits):
                    self.encodings[categories[category_idx]].append(bit)
                    if bit not in self.bits:
                        self.bits[bit] = {categories[category_idx]}
                    else:
                        self.bits[bit].add(categories[category_idx])

                start_bit += step

            # wrap around to first categories
            #
            self.encodings[categories[-1]] = []
            for bit in range(start_bit, self.encodings[categories[-2]][-1] + 1):
                self.encodings[categories[-1]].append(bit)
                if bit not in self.bits:
                    self.bits[bit] = {categories[-1]}
                else:
                    self.bits[bit].add(categories[-1])

            for bit in range(0, self.n_bits - len(self.encodings[categories[-1]])):
                self.encodings[categories[-1]].append(bit)
                if bit not in self.bits:
                    self.bits[bit] = {categories[-1]}
                else:
                    self.bits[bit].add(categories[-1])

        else:
            for category_idx in range(len(categories)):
                self.encodings[categories[category_idx]] = set()
                for bit in range(start_bit, start_bit + self.n_bits):
                    self.encodings[categories[category_idx]].add(bit)
                    if bit not in self.bits:
                        self.bits[bit] = {categories[category_idx]}
                    else:
                        self.bits[bit].add(categories[category_idx])

                start_bit += step

    def encode(self, category) -> set:

        # if the category already exists then retrieve
        #
        if category in self.encodings:
            enc = set(self.encodings[category])
        else:
            # if not then assume not a semantic category so create random distribution
            #

            # set the state of the random generator
            #
            random.setstate(Encoder.rand_states[self.name])

            # calculate the bit population once
            #
            offset = self.bit_offset * self.enc_size
            bit_population = [i for i in range(offset, offset + self.enc_size)]

            # get the encoded bits
            #
            enc = set(random.sample(population=bit_population, k=self.n_bits))

            Encoder.rand_states[self.name] = random.getstate()

            self.encodings[category] = set(enc)

            # maintain the mapping of bits to bucket
            #
            for bit in enc:
                if bit not in self.bits:
                    self.bits[bit] = {category}
                else:
                    self.bits[bit].add(category)

        return enc

    def decode(self, enc) -> list:
        categories = {}

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

        # sum the weights for the categories associated with the bits in the encoding
        #
        total_weight = 0.0
        for bit in enc:
            # only process bits from this encoder
            #
            if bit in self.bits:
                for category in self.bits[bit]:
                    if category not in categories:
                        categories[category] = enc[bit]
                    else:
                        categories[category] += enc[bit]
                    total_weight += enc[bit]

        categories = [(category, categories[category] / self.n_bits) for category in categories]

        categories.sort(key=lambda x: x[1], reverse=True)

        return categories


if __name__ == '__main__':

    categories = [c for c in range(20)]
    encoder_1 = CategoryEncoder(name='semantic_1',
                                semantic_categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=False)

    enc_1 = encoder_1.encode(category=0)

    decode_1 = encoder_1.decode(enc=enc_1)

    encoder_2 = CategoryEncoder(name='semantic_2',
                                semantic_categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=True)

    enc_2 = encoder_2.encode(category=0)

    decode_2 = encoder_2.decode(enc=enc_2)

    print('finished')
