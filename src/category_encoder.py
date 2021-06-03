#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from typing import Optional, Union
import cython


@cython.cclass
class CategoryEncoder(object):
    """
    CategoryEncoder class can encode either a single category (string) or a list of related categories, where the order of the list implies a level of similarity between neighbours
    """

    # ***************************************
    # cython declarations for the class attributes
    #
    type = cython.declare(str, visibility='public')
    name = cython.declare(str, visibility='public')
    encodings = cython.declare(dict, visibility='public')
    bits = cython.declare(dict, visibility='public')
    n_bits = cython.declare(cython.int, visibility='public')
    enc_size = cython.declare(cython.int, visibility='public')
    bit_offset = cython.declare(cython.int, visibility='public')
    periodic = cython.declare(cython.bint, visibility='public')
    semantic = cython.declare(cython.bint, visibility='public')
    rand_state = cython.declare(object, visibility='public')

    def __init__(self,
                 name='category',
                 semantic_categories: list = None,
                 n_bits: cython.int = 40,
                 enc_size: cython.int = 2048,
                 bit_offset: cython.int = 0,
                 periodic: cython.bint = False,
                 seed: cython.int = 12345):
        """
        Creates a CategoryEncoder
        :param name: different instances of encoders can be identified by a given name.
        :param semantic_categories: defines the list of categories that can be encoded by this encoder. Neighbours within the list will have a level of similarity related
                                    to the number of bits in the encoding.
        :param n_bits: the number of bits in the encoding - this should be at least 20 and should be roughly 2% (ie sparsity of 2%) of the enc_size
        :param enc_size: the total possible number of bits in the sparse encoding - this should ensure a sparsity of around 2%
        :param bit_offset: this factor offsets the starting bit - set a 0 means bits start from 0 up to enc_size-1. Set to 1 and bits start from enc_size and range up to (enc_size * 2) - 1
        :param periodic: boolean flag applies only to a category list and ensures the first category entry is related to the last category entry
        :param seed: each encoder instance can have a different random seed to provide deterministic sequence of random bits
        """

        self.type = 'category'
        """ indicates the type of encoder """

        self.name = name
        """ the specific instance name of this encoder """

        self.encodings = {}
        """ a map between the category and set of bits that represent its encoding """

        self.bits = {}
        """ a map between each bit and the category encodings it is a part of """

        self.n_bits = n_bits
        """ the maximum number of ON bits an encoding consists of """

        self.enc_size = enc_size
        """ the maximum possible bits an encoding a sparse encoding can be created from """

        self.bit_offset = bit_offset
        """ a factor of enc_size that defines the starting bit number """

        # seed the generator if required
        #
        if seed is not None:
            random.seed(seed)

        self.rand_state = random.getstate()
        """ the state of the random number generator for this encoder """

        self.periodic = periodic
        """ flag applies to category list and ensures the catgeories at the start and end of the list are similar """

        self.semantic = False
        """ flag indicating that the categories have semantic similarity """

        # encode the semantic list of categories
        #
        if semantic_categories is not None:
            self.set_semantic_categories(categories=semantic_categories)

    def set_semantic_categories(self, categories: list):
        """
        Encodes a list of semantically related categories. If the Encoder is configured to be periodic the categories and the start and end of the lists will be semantically similar.
        The algorithm ensures there is at least 0.5 * n_bits of overlap between each neighbouring category
        :param categories: the semantically related list of categories
        :return: None
        """

        # ************************
        # help cython type variables
        #
        step: cython.int
        start_bit: cython.int
        category_idx: cython.int
        bit: cython.int

        # ensure this encoder is configured as a semantic encoder
        #
        self.semantic = True

        # calculate the number of bit difference between each category encoding
        #
        if len(categories) < self.n_bits and self.periodic:
            # ensure only the immediate neighbours in the category list are semantically similar
            #
            step = int(self.n_bits / 2)
        else:
            # as the number of categories is >= to the number of bits then there can be 1 bit difference between each neighbouring categeory
            step = 1

        # start the bit encoding from the required offset
        #
        start_bit = self.bit_offset * self.enc_size

        # if a periodic encoding is required then the first and last categories must be related
        #
        if self.periodic:

            # encode the first to last - 1 categories first
            #
            for category_idx in range(len(categories) - 1):

                self.encodings[categories[category_idx]] = []
                for bit in range(start_bit, start_bit + self.n_bits):

                    # map the category to its bits
                    #
                    self.encodings[categories[category_idx]].append(bit)

                    # map the bit to its categories
                    #
                    if bit not in self.bits:
                        self.bits[bit] = {categories[category_idx]}
                    else:
                        self.bits[bit].add(categories[category_idx])

                # setup the start bit for the next category ensuring there is an overlap - as step will always be less than n_bits
                #
                start_bit += step

            # overlap the last category with its preceding neighbour
            #
            self.encodings[categories[-1]] = []
            for bit in range(start_bit, self.encodings[categories[-2]][-1] + 1):
                self.encodings[categories[-1]].append(bit)
                if bit not in self.bits:
                    self.bits[bit] = {categories[-1]}
                else:
                    self.bits[bit].add(categories[-1])

            # overlap the last category with the first categories
            #
            for bit in range(self.bit_offset * self.enc_size, self.n_bits - len(self.encodings[categories[-1]])):
                self.encodings[categories[-1]].append(bit)
                if bit not in self.bits:
                    self.bits[bit] = {categories[-1]}
                else:
                    self.bits[bit].add(categories[-1])

        # encoder not periodic so no need to ensure last category is similar to first category
        #
        else:

            for category_idx in range(len(categories)):
                self.encodings[categories[category_idx]] = set()
                for bit in range(start_bit, start_bit + self.n_bits):

                    # map the category to its bits
                    #
                    self.encodings[categories[category_idx]].add(bit)

                    # map the bit to its categories
                    #
                    if bit not in self.bits:
                        self.bits[bit] = {categories[category_idx]}
                    else:
                        self.bits[bit].add(categories[category_idx])

                # ensure each category has an overlap with the preceding category as step is always less that n_bits
                #
                start_bit += step

    def encode(self, category: Optional[str]) -> set:
        """
        encodes a category - if the encoder is a semantic encoder then it can only encode categories in the configured list. If not a semantic encoder then each category encoding has
        negligible overlap
        :param category: category to encode
        :return: a set of bits that encode the category
        """

        # ************************
        # help cython type variables
        #
        enc: Optional[set] = None
        offset: cython.int
        bit_population: list
        bit: cython.int

        # if the category already exists then retrieve
        #
        if self.semantic:
            if category in self.encodings:
                enc = set(self.encodings[category])
        else:
            if category in self.encodings:
                enc = set(self.encodings[category])

            # category not already encoded and this is not a semantic category so create random distribution
            #
            else:

                # set the state of the random generator
                #
                random.setstate(self.rand_state)

                # calculate the bit population once
                #
                offset = self.bit_offset * self.enc_size
                bit_population = [bit for bit in range(offset, offset + self.enc_size)]

                # get the set of encoded bits
                #
                enc = set(random.sample(population=bit_population, k=self.n_bits))

                # remember the random state for next time
                #
                self.rand_state = random.getstate()

                # map the category to a copy of the set of encoded bits
                #
                self.encodings[category] = set(enc)

                # maintain the mapping of bits to category
                #
                for bit in enc:
                    if bit not in self.bits:
                        self.bits[bit] = {category}
                    else:
                        self.bits[bit].add(category)

        return enc

    def decode(self, enc: Union[set, dict], max_bit_weight: float = 1.0) -> list:
        """
        decodes a set of bits into categories
        :param enc: can be either a set of bits or a dictionary of bits in which the bits are the keys and the values are the weight of each bit
        :param max_bit_weight: the maximum a bit weight can be
        :return: a list of decoded categories with associated weights
        """

        # ************************
        # help cython type variables
        #
        bit: cython.int
        total_weight: float
        category: str
        categories: dict = {}
        category_list: list
        weight_adj: float

        # add default weights of 1.0 if given a set of bits
        #
        if isinstance(enc, set):
            enc = {bit: max_bit_weight for bit in enc}

        # sum the weights for the categories associated with the bits in the encoding
        #
        total_weight = 0.0
        for bit in enc:
            if bit in self.bits:
                for category in self.bits[bit]:
                    if category not in categories:
                        categories[category] = enc[bit]
                    else:
                        categories[category] += enc[bit]
                    total_weight += enc[bit]

        # create a list of categories - if not semantic then filter out those whose weights are less than or equal to 2 * max_bit_weight -
        # which amounts to encoding noise
        #
        weight_adj = self.n_bits * max_bit_weight
        category_list = [(category, categories[category] / weight_adj)
                         for category in categories
                         if self.semantic or categories[category] > 2 * max_bit_weight]

        category_list.sort(key=lambda x: x[1], reverse=True)

        return category_list


if __name__ == '__main__':

    categories = [str(c) for c in range(20)]
    encoder_1 = CategoryEncoder(name='semantic_1',
                                semantic_categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=False)

    enc_1 = encoder_1.encode(category='0')

    decode_1 = encoder_1.decode(enc=enc_1)

    encoder_2 = CategoryEncoder(name='semantic_2',
                                semantic_categories=categories,
                                n_bits=40,
                                enc_size=2048,
                                periodic=True)

    enc_2 = encoder_2.encode(category='0')

    decode_2 = encoder_2.decode(enc=enc_2)

    encoder_3 = CategoryEncoder(name='non_semantic',
                                semantic_categories=None,
                                n_bits=40,
                                enc_size=2048,
                                periodic=False)

    enc_3_0 = encoder_3.encode(category='0')
    enc_3_1 = encoder_3.encode(category='1')
    enc_3_2 = encoder_3.encode(category='2')

    decode_3_0 = encoder_3.decode(enc=enc_3_0)
    decode_3_1 = encoder_3.decode(enc=enc_3_1)
    decode_3_2 = encoder_3.decode(enc=enc_3_2)

    print('finished')
