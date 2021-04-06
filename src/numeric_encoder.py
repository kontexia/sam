#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random
from src.encoder import Encoder


class NumericEncoder(Encoder):
    def __init__(self, min_step: float = 1.0, n_bits: int = 40, enc_size: int = 2048, seed=12345):
        Encoder.__init__(self, encoder_type='numeric', n_bits=n_bits, enc_size=enc_size, seed=seed)

        self.min_step = min_step

        self.min_seen: int = 0
        self.min: int = 0
        self.min_next_idx: int = 0

        self.max_seen = 0
        self.max: int = 0
        self.max_next_idx: int = 0
        self.zero_bucket = None
        random.seed(seed)

    def create_encoding(self, target_bucket):

        # set the state of the random generator
        #
        random.setstate(Encoder.rand_states['numeric'])

        if target_bucket > self.max:

            # will need the bits from the current largest bucket encoding
            #
            prev_enc = self.encodings[self.max]
            for bucket in range(self.max + 1, target_bucket + self.n_bits + 1):

                # create the new encoding as a copy of the last max bucket
                #
                new_enc = [i for i in prev_enc]

                # get another bit chosen at random that's not in previous bucket encoding
                #
                new_bit = random.sample(population=[i for i in range(self.enc_size) if i not in prev_enc], k=1)

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
                self.max = bucket

        # else must be below minimum
        #
        else:
            prev_enc = self.encodings[self.min]
            for bucket in range(self.min - 1, target_bucket - self.n_bits - 1, -1):

                # create the new encoding as a copy of the last max bucket
                #
                new_enc = [i for i in prev_enc]

                # get another bit chosen at random that's not in previous bucket encoding
                #
                new_bit = random.sample(population=[i for i in range(self.enc_size) if i not in prev_enc], k=1)

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
                self.min = bucket

        # remember the state of the random generator
        #
        Encoder.rand_states['numeric'] = random.getstate()

    def encode(self, numeric) -> set:

        # round the numeric to the minimum step
        #
        round_numeric = round(numeric / self.min_step)

        # if no existing encodings then create first one
        #
        if len(self.encodings) == 0:

            bucket = 0

            # setup up the zero bucket association with real number
            #
            self.zero_bucket = round_numeric

            # set the state of the random generator
            #
            random.setstate(Encoder.rand_states['numeric'])

            # create a list of random numbers to represent the bits set
            #
            new_enc = list(random.sample(population=range(self.enc_size), k=self.n_bits))

            Encoder.rand_states['numeric'] = random.getstate()

            self.encodings[bucket] = new_enc

            # the max bucket that exists along with the next offset in list of bits to change
            #
            self.max = bucket
            self.max_next_idx = 0

            # the min bucket that exists along with the next offset in list of bits to change
            #
            self.min = bucket
            self.min_next_idx = 0

            # maintain the mapping of bits to bucket to allow for easy decoding
            #
            for bit in new_enc:
                if bit not in self.bits:
                    self.bits[bit] = {bucket}
                else:
                    self.bits[bit].add(bucket)

            enc = set(self.encodings[bucket])

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

            best_weight = buckets[0][1]
            if best_weight < self.n_bits:
                value = 0.0
                total_weight = 0.0

                #for idx in range(len(buckets)):
                for idx in range(min(3, len(buckets))):
                    value += (buckets[idx][0] + self.zero_bucket) * self.min_step * buckets[idx][1]
                    total_weight += buckets[idx][1]

                value = value / total_weight
            else:
                value = (buckets[0][0] * self.min_step) + (self.zero_bucket * self.min_step)
        else:
            value = None
        return value


if __name__ == '__main__':

    from src.sdr import SDR

    encoder = NumericEncoder(min_step=0.5, n_bits=40, enc_size=2048)

    enc_1 = encoder.encode(100)
    enc_4 = encoder.encode(120.0)
    enc_5 = encoder.encode(99)

    val_1 = encoder.decode(enc_1)

    enc_2 = encoder.encode(102)

    val_2 = encoder.decode(enc_2)

    enc_3 = encoder.encode(100.5)
    val_3 = encoder.decode(enc_3)

    val_4 = encoder.decode(enc_4)

    sdr_1 = SDR(enc_type='test', value=100, encoder=encoder)
    sdr_2 = SDR(enc_type='test', value=102, encoder=encoder)
    sdr_3 = SDR(enc_type='test', value=100.5, encoder=encoder)

    d_1_2 = sdr_1.distance(sdr_2)

    d_1_3 = sdr_1.distance(sdr_3)

    learn_rate = 0.5
    sdr_1.learn(sdr=sdr_2, learn_rate=learn_rate, prune=0.01)
    d_1_4 = sdr_1.distance(sdr_2)

    val_1_learned_1 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=learn_rate, prune=0.01)
    d_1_5 = sdr_1.distance(sdr_2)

    val_1_learned_2 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=learn_rate, prune=0.01)
    d_1_6 = sdr_1.distance(sdr_2)

    val_1_learned_3 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=learn_rate, prune=0.01)
    d_1_7 = sdr_1.distance(sdr_2)

    val_1_learned_4 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=learn_rate, prune=0.01)
    d_1_8 = sdr_1.distance(sdr_2)

    val_1_learned_5 = sdr_1.decode()

    print('finished')
