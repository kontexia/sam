#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.numeric_encoder import NumericEncoder
from src.category_encoder import CategoryEncoder


class ValueEncoder(object):
    def __init__(self, name: str,
                 n_bits: int = 40,
                 enc_size: int = 2048,
                 numeric_step: float = 1.0):
        self.name = name
        self.n_bits = n_bits
        self.enc_size = enc_size
        self.type = 'ValueEncoder'

        self.encoders = {'strlen': NumericEncoder(name=f'{self.name}_strlen',
                                                  n_bits=self.n_bits,
                                                  enc_size=self.enc_size,
                                                  min_step=1.0,
                                                  bit_offset=1),
                         'string': CategoryEncoder(name=f'{self.name}_string',
                                                   n_bits=self.n_bits,
                                                   enc_size=self.enc_size,
                                                   bit_offset=4),
                         'numeric': NumericEncoder(name=f'{self.name}_numeric',
                                                   n_bits=self.n_bits,
                                                   enc_size=self.enc_size,
                                                   min_step=numeric_step,
                                                   bit_offset=2)
                         }

    def encode(self, value) -> set:
        if value is None:

            # string length
            #
            enc = self.encoders['strlen'].encode(numeric=0)

            # string value
            #
            enc.update(self.encoders['string'].encode(category=None))

        elif isinstance(value, int) or isinstance(value, float):
            # string length of 0
            #
            enc = self.encoders['strlen'].encode(numeric=0)

            # the value
            #
            enc.update(self.encoders['numeric'].encode(numeric=value))

        else:
            # string length
            #
            enc = self.encoders['strlen'].encode(numeric=len(value))

            # string value
            #
            enc.update(self.encoders['string'].encode(category=value))
        return enc

    def decode(self, enc):

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

        decoding = dict()
        decoding['strlen'] = self.encoders['strlen'].decode(enc=enc)
        if decoding['strlen'] == 0:
            decoding['value'] = self.encoders['numeric'].decode(enc=enc)
        else:
            decoding['value'] = self.encoders['string'].decode(enc=enc)
        return decoding


if __name__ == '__main__':

    encoder = ValueEncoder(name='test',
                           n_bits=10,
                           enc_size=2048,
                           numeric_step=1)

    enc_1 = encoder.encode(value=100)
    enc_2 = encoder.encode(value=120)

    enc_3 = encoder.encode(value="hello")

    enc_4 = encoder.encode(value=None)

    dec_1 = encoder.decode(enc=enc_1)
    dec_2 = encoder.decode(enc=enc_2)
    dec_3 = encoder.decode(enc=enc_3)
    dec_4 = encoder.decode(enc=enc_4)

    print('finished')
