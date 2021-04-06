#!/usr/bin/env python
# -*- encoding: utf-8 -*-


class CategoryEncoder(object):
    def __init__(self, categories=None, n_bits: int = 40, enc_size: int = 2048, periodic=False):
        self.encodings = {}
        self.bits = {}
        self.periodic = periodic
        self.n_bits = n_bits
        self.enc_size = enc_size
        if categories is not None:
            self.set_categories(categories=categories)

    def set_categories(self, categories):

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
        if category in self.encodings:
            enc = set(self.encodings[category])
        else:
            enc = set()
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
            for category in self.bits[bit]:
                if category not in categories:
                    categories[category] = enc[bit]
                else:
                    categories[category] += enc[bit]
                total_weight += enc[bit]

        categories = [(category, categories[category] / total_weight) for category in categories]
        categories.sort(key=lambda x: x[1], reverse=True)

        return categories


if __name__ == '__main__':

    categories = [c for c in range(20)]
    encoder_1 = CategoryEncoder(categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=False)

    enc_1 = encoder_1.encode(category=0)

    decode_1 = encoder_1.decode(enc=enc_1)

    encoder_2 = CategoryEncoder(categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=True)

    enc_2 = encoder_2.encode(category=0)

    decode_2 = encoder_2.decode(enc=enc_2)

    print('finished')