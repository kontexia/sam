#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cython
from typing import Optional, Union
import random
from math import log10, pow


@cython.cclass
class NumericEncoder(object):
    """
    NumericEncoder encodes both integers and floats and ensures that those numerics close to each other on the number line have a certain level of similarity
    The encoder employs a random distribution algorithm that requires state (knowledge of previously encoded numerics) in order to guarantee the correct level
    of similarity
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
    rand_state = cython.declare(object, visibility='public')

    log = cython.declare(cython.bint, visibility='public')
    min_step = cython.declare(float, visibility='public')
    min_bucket = cython.declare(cython.int, visibility='public')
    min_next_idx = cython.declare(cython.int, visibility='public')
    max_bucket = cython.declare(cython.int, visibility='public')
    max_next_idx = cython.declare(cython.int, visibility='public')
    zero_bucket = cython.declare(float, visibility='public')

    def __init__(self,
                 name: str = 'numeric',
                 min_step: float = 1.0,
                 n_bits: cython.int = 40,
                 enc_size: cython.int = 2048,
                 bit_offset: cython.int = 0,
                 log: cython.bint = False,
                 seed=12345):
        """
        Creates a Numeric Encoder using either a linear or log random distributed algorithm
        :param name: different instances of encoders can be identified by a given name.
        :param min_step: the minimum step amount that can be encoded. For example if set to 1.0 the encoder encodes to the nearest integer
        :param n_bits: the number of bits in the encoding - this should be at least 20 and should be roughly 2% (ie sparsity of 2%) of the enc_size
        :param enc_size: the total possible number of bits in the sparse encoding - this should ensure a sparsity of around 2%
        :param bit_offset: this factor offsets the starting bit - set a 0 means bits start from 0 up to enc_size-1. Set to 1 and bits start from enc_size and range up to (enc_size * 2) - 1
        :param log: flag to indicate if the log of the numeric is encoded - if set to true then the min_step applies to a log scale - meaning that
                    values such as 1.0 and 2.0 will have an encoded similarity similar to 1000 and 2000
        :param seed: each encoder instance can have a different random seed to provide deterministic sequence of random bits
        """

        self.type = 'numeric'
        """ indicates the type of encoder """

        self.name = name
        """ the specific instance name of this encoder """

        self.encodings = {}
        """ a map between the bucket and set of bits that represent its encoding """

        self.bits = {}
        """ a map between each bit and the bucket encodings it is a part of """

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

        self.log: cython.bint = log
        """ flag to indicate if the numeric is transformed by the log function """

        self.zero_bucket: float
        """ the numeric value that bucket 0 represents """

        self.min_step: float = min_step
        """ the minimum numerical amount that can be encoded"""

        self.min_bucket: cython.int = 0
        """ the current minimum bucket that has been encoded """

        self.min_next_idx: cython.int = 0
        """ the next bit index (a number between 0 and n_bits) into the minimum bucket encoding that will be replaced with a random bit """

        self.max_bucket: cython.int = 0
        """ the current maximum bucket that has been encoded """

        self.max_next_idx: cython.int = 0
        """ the next bit index (a number between 0 and n_bits) into the maximum bucket encoding that will be replaced with a random bit """

    def create_encoding(self, target_bucket: Optional[int]):
        """
        creates encodings between the current min or max_bucket up to and including the target bucket
        :param target_bucket: the target bucket to encode
        :return: None
        """

        # ************************
        # help cython type variables
        #
        offset: cython.int
        bit: cython.int
        bit_population: list
        new_enc: list
        prev_enc: list
        bucket: cython.int

        # set the state of the random generator
        #
        random.setstate(self.rand_state)

        # calculate the bit population, offset as required
        #
        offset = self.bit_offset * self.enc_size
        bit_population = [bit for bit in range(offset, offset + self.enc_size)]

        # the value none will have a specific value
        #
        if target_bucket is None:

            # create a list of random numbers to represent the bits set
            #
            new_enc = list(random.sample(population=bit_population, k=self.n_bits))

            # map the bucket to the encoding
            #
            self.encodings[target_bucket] = new_enc

            # maintain the mapping of bits to bucket to allow for easy decoding
            #
            for bit in new_enc:
                if bit not in self.bits:
                    self.bits[bit] = {target_bucket}
                else:
                    self.bits[bit].add(target_bucket)

        # the first encoded value is special
        #
        elif target_bucket == 0:

            # create a list of random numbers to represent the bits set
            #
            new_enc = list(random.sample(population=bit_population, k=self.n_bits))

            # map the 0 bucket to the encoding
            #
            self.encodings[target_bucket] = new_enc

            # set the max bucket that exists along with the next offset in list of bits to change
            #
            self.max_bucket = target_bucket
            self.max_next_idx = 0

            # set the min bucket that exists along with the next offset in list of bits to change
            #
            self.min_bucket = target_bucket
            self.min_next_idx = 0

            # maintain the mapping of bits to bucket to allow for easy decoding
            #
            for bit in new_enc:
                if bit not in self.bits:
                    self.bits[bit] = {target_bucket}
                else:
                    self.bits[bit].add(target_bucket)

        # if target bucket is larger than current max_bucket so fill in the gaps
        #
        elif target_bucket > self.max_bucket:

            # will need the bits from the current largest bucket encoding
            #
            prev_enc = self.encodings[self.max_bucket]

            # from the current max bucket + 1 up to and including the target_bucket
            #
            for bucket in range(self.max_bucket + 1, target_bucket + self.n_bits + 1):

                # create the new encoding as a copy of the last max bucket
                #
                new_enc = [bit for bit in prev_enc]

                # get another bit chosen at random that's not in previous bucket encoding
                #
                new_bit = random.sample(population=[bit for bit in bit_population if bit not in prev_enc], k=1)

                # replace one bit at the max_next_idx slot which guarantees no clashes
                #
                new_enc[self.max_next_idx] = new_bit[0]

                # update the next idx to replace, remembering to wrap around if necessary
                #
                self.max_next_idx += 1
                if self.max_next_idx >= self.n_bits:
                    self.max_next_idx = 0

                # save new encoding
                #
                self.encodings[bucket] = new_enc

                # maintain the mapping of bits to buckets
                #
                for bit in new_enc:
                    if bit not in self.bits:
                        self.bits[bit] = {bucket}
                    else:
                        self.bits[bit].add(bucket)

                # remember the previous encoding
                #
                prev_enc = new_enc

                # we now have a new max bucket
                #
                self.max_bucket = bucket

        # else must be below minimum
        #
        else:
            prev_enc = self.encodings[self.min_bucket]
            for bucket in range(self.min_bucket - 1, target_bucket - self.n_bits - 1, -1):

                # create the new encoding as a copy of the last max bucket
                #
                new_enc = [i for i in prev_enc]

                # get another bit chosen at random that's not in previous bucket encoding
                #
                new_bit = random.sample(population=[i for i in bit_population if i not in prev_enc], k=1)

                # replace one bit at the max_next_idx slot which guarantees no clashes
                #
                new_enc[self.min_next_idx] = new_bit[0]

                # update the next idx to replace, remembering to wrap around if necessary
                #
                self.min_next_idx += 1
                if self.min_next_idx >= self.n_bits:
                    self.min_next_idx = 0

                # save new encoding
                #
                self.encodings[bucket] = new_enc

                # maintain the mapping of bits to buckets
                #
                for bit in new_enc:
                    if bit not in self.bits:
                        self.bits[bit] = {bucket}
                    else:
                        self.bits[bit].add(bucket)

                # remember the previous encoding
                #
                prev_enc = new_enc

                # we now have a new min bucket
                #
                self.min_bucket = bucket

        # remember the state of the random generator
        #
        self.rand_state = random.getstate()

    def encode(self, numeric: Optional[float]) -> set:
        """
        encodes a numeric
        :param numeric: the numeric to encode - can be a None
        :return: a set of encoded bits
        """

        # ************************
        # help cython type variables
        #
        enc: set
        round_numeric: float
        target_bucket: cython.int

        # if its none then create special encoding that has no similarity to other numbers
        #
        if numeric is None:

            # if None hasn't already been encoded then encode
            #
            if None not in self.encodings:
                self.create_encoding(None)

            enc = set(self.encodings[numeric])

        # else assume its a numeric
        #
        else:

            # if configured to use the log then take the log to base 10 of the ration of the numeric and min step
            # need to cater for numerics < 1 as cannot take log of numbers less than 1
            #
            if self.log:
                if numeric >= 1.0:
                    round_numeric = round(log10(numeric) / self.min_step)
                elif numeric <= -1.0:
                    round_numeric = -1 * round(log10(numeric) / self.min_step)
                else:
                    round_numeric = round(numeric / self.min_step)

            # round the numeric to the minimum step
            #
            else:
                round_numeric = round(numeric / self.min_step)

            # if no existing encodings then create first one for the zero bucket
            #
            if len(self.encodings) == 0 or (len(self.encodings) == 1 and None in self.encodings):

                # setup up the zero bucket association with real number
                #
                self.zero_bucket = round_numeric

                # create the 0 bucket encoding
                #
                self.create_encoding(0)

                # remember the encoding required
                #
                enc = set(self.encodings[0])

                # also create preceding and succeeding buckets
                #
                self.create_encoding(1)
                self.create_encoding(-1)
            else:

                # calculate the bucket associated with the encoding
                #
                target_bucket = int(round_numeric - self.zero_bucket)

                # just return encoding if bucket exists
                #
                if target_bucket in self.encodings:
                    enc = set(self.encodings[target_bucket])
                else:

                    self.create_encoding(target_bucket)

                    # the encoding required
                    #
                    enc = set(self.encodings[target_bucket])

        return enc

    def decode(self, enc: Union[set, dict], max_bit_weight: float = 1.0):
        """
        decodes a set of bits into a numeric
        :param enc: can be either a set of bits or a dictionary of bits in which the bits are the keys and the values are the weight of each bit
        :param max_bit_weight: the maximum a bit weight can be
        :return: a weighted average numeric - where the wights are based on the bit weights
        """

        # ************************
        # help cython type variables
        #
        buckets: dict = {}
        bucket_list: list
        bit: cython.int
        bucket: cython.int
        n: cython.int
        best_weight: float
        total_weight: float
        idx: cython.int

        # add default weights of 1.0 if given a set of bits
        #
        if isinstance(enc, set):
            enc = {bit: max_bit_weight for bit in enc}

        # sum the weights for the buckets associated with the bits in the encoding
        #
        for bit in enc:
            # only process bits for this encoder
            #
            if bit in self.bits:
                for bucket in self.bits[bit]:
                    if bucket not in buckets:
                        buckets[bucket] = enc[bit]
                    else:
                        buckets[bucket] += enc[bit]

        if len(buckets) > 0:

            # create a list of buckets so we can sort in descending order of bit weight
            #
            bucket_list = [(n, buckets[n] / max_bit_weight) for n in buckets]
            bucket_list.sort(key=lambda x: x[1], reverse=True)

            # get weighted average of bucket values if the best weight is less than n_bits
            #
            best_weight = bucket_list[0][1]
            if best_weight < self.n_bits:

                value = None
                total_weight = 0.0

                # look only at the top 3 buckets which should contain the most relevant values
                # if any of these are None then return None
                #
                for idx in range(min(3, len(bucket_list))):
                    if bucket_list[idx][0] is not None:
                        if value is None:
                            value = 0
                        value += (bucket_list[idx][0] + self.zero_bucket) * self.min_step * bucket_list[idx][1]
                        total_weight += bucket_list[idx][1]

                if value is not None:
                    value = value / total_weight

                    # return to linear scale if log previously applied
                    #
                    if self.log:
                        value = pow(10, value)
            # weight of best bucket is a maximum of n_bits so don't need to calculate a weighted average
            #
            else:

                # it's possible the bucket value is actually None
                #
                if bucket_list[0][0] is not None:

                    # calc value of best bucket using min_step and offset from zero bucket value
                    #
                    value = (bucket_list[0][0] * self.min_step) + (self.zero_bucket * self.min_step)

                    # return to linear scale if log previously applied
                    #
                    if self.log:
                        value = pow(10, value)
                else:
                    value = None
        else:
            value = None
        return value


if __name__ == '__main__':

    encoder = NumericEncoder(min_step=0.5, n_bits=40, enc_size=2048, bit_offset=1)

    enc_n = encoder.encode(None)
    val_n = encoder.decode(enc_n)

    enc_n = encoder.encode(None)

    enc_1 = encoder.encode(100)
    enc_4 = encoder.encode(120.0)
    enc_5 = encoder.encode(99)

    val_1 = encoder.decode(enc_1)

    enc_2 = encoder.encode(102)

    val_2 = encoder.decode(enc_2)

    enc_3 = encoder.encode(100.5)
    val_3 = encoder.decode(enc_3)

    val_4 = encoder.decode(enc_4)

    print('finished')
