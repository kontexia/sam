#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from src.encoder import Encoder
from math import log10, pow


class NumericEncoder(Encoder):
    def __init__(self,
                 name: str = 'numeric',
                 min_step: float = 1.0,
                 n_bits: int = 40,
                 enc_size: int = 2048,
                 bit_offset: int = 0,
                 log: bool = False,
                 seed=12345):
        Encoder.__init__(self, encoder_type='numeric', name=name, n_bits=n_bits, enc_size=enc_size, bit_offset=bit_offset, seed=seed)

        self.log = log
        self.min_step = min_step

        self.min_bucket: int = 0
        self.min_next_idx: int = 0

        self.max_bucket: int = 0
        self.max_next_idx: int = 0
        self.zero_bucket = None

        random.seed(seed)

    def create_encoding(self, target_bucket):

        # set the state of the random generator
        #
        random.setstate(Encoder.rand_states[self.name])

        # calculate the bit population once
        #
        offset = self.bit_offset * self.enc_size
        bit_population = [i for i in range(offset, offset + self.enc_size)]

        # the value none will have a specific value
        #
        if target_bucket is None:

            # create a list of random numbers to represent the bits set
            #
            new_enc = list(random.sample(population=bit_population, k=self.n_bits))

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
            self.encodings[target_bucket] = new_enc

            # the max bucket that exists along with the next offset in list of bits to change
            #
            self.max_bucket = target_bucket
            self.max_next_idx = 0

            # the min bucket that exists along with the next offset in list of bits to change
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

        elif target_bucket > self.max_bucket:

            # will need the bits from the current largest bucket encoding
            #
            prev_enc = self.encodings[self.max_bucket]
            for bucket in range(self.max_bucket + 1, target_bucket + self.n_bits + 1):

                # create the new encoding as a copy of the last max bucket
                #
                new_enc = [i for i in prev_enc]

                # get another bit chosen at random that's not in previous bucket encoding
                #
                new_bit = random.sample(population=[i for i in bit_population if i not in prev_enc], k=1)

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
        Encoder.rand_states[self.name] = random.getstate()

    def encode(self, numeric) -> set:

        # if its none then create special encoding that has no similarity to other numbers
        #
        if numeric is None:

            if None not in self.encodings:
                self.create_encoding(target_bucket=None)

            enc = set(self.encodings[numeric])

        # assume its a numeric
        #
        else:

            # round the numeric to the minimum step
            #
            if self.log:
                if numeric >= 1.0:
                    round_numeric = round(log10(numeric) / self.min_step)
                elif numeric <= -1.0:
                    round_numeric = -1 * round(log10(numeric) / self.min_step)
                else:
                    round_numeric = round(numeric / self.min_step)
            else:
                round_numeric = round(numeric / self.min_step)

            # if no existing encodings then create first one for the zero bucket
            #
            if self.zero_bucket is None:

                # setup up the zero bucket association with real number
                #
                self.zero_bucket = round_numeric

                # create the 0 bucket encoding
                #
                self.create_encoding(target_bucket=0)

                # remember the encoding required
                #
                enc = set(self.encodings[0])

                # also create preceding and succeeding buckets
                #
                self.create_encoding(target_bucket=1)
                self.create_encoding(target_bucket=-1)
            else:

                # calculate the bucket associated with the encoding
                #
                target_bucket = round_numeric - self.zero_bucket

                # just return encoding if bucket exists
                #
                if target_bucket in self.encodings:
                    enc = set(self.encodings[target_bucket])
                else:

                    self.create_encoding(target_bucket=target_bucket)

                    # the encoding required
                    #
                    enc = set(self.encodings[target_bucket])

        return enc

    def decode(self, enc):
        buckets = {}

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

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
            # create a list of buckets so we can sort
            #
            buckets = [(n, buckets[n]) for n in buckets]
            buckets.sort(key=lambda x: x[1], reverse=True)

            # get weighted average of bucket value if not None
            best_weight = buckets[0][1]
            if best_weight < self.n_bits:
                value = None
                total_weight = 0.0

                # look only at the top 3 buckets which should contain the most relevant values
                # if any of these are None then return None
                for idx in range(min(3, len(buckets))):
                    if buckets[idx][0] is not None:
                        if value is None:
                            value = 0
                        value += (buckets[idx][0] + self.zero_bucket) * self.min_step * buckets[idx][1]
                        total_weight += buckets[idx][1]

                if value is not None:
                    value = value / total_weight
                    if self.log:
                        value = pow(10, value)
            else:
                if buckets[0][0] is not None:
                    value = (buckets[0][0] * self.min_step) + (self.zero_bucket * self.min_step)
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
