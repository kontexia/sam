#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.sgm import SGM
from src.numeric_encoder import NumericEncoder
from src.string_encoder import StringEncoder

import time


if __name__ == '__main__':

    n_encoder = NumericEncoder(min_step=1,
                               n_bits=40,
                               enc_size=2048,
                               seed=12345)

    sgm_11 = SGM()
    sgm_11.add_encoding(enc_type='test', value=100, encoder=n_encoder)
    sgm_22 = SGM()
    sgm_22.add_encoding(enc_type='test', value=100, encoder=n_encoder)

    s_1 = sgm_11.similarity(sgm_22)
    sgm_33 = SGM()
    sgm_33.add_encoding(enc_type='test', value=120, encoder=n_encoder)
    s_2 = sgm_11.similarity(sgm_33)

    sgm_1 = SGM()
    for idx in range(10):
        sgm_1.add_encoding(enc_type=str(idx), value=100, encoder=n_encoder)

    sgm_2 = SGM()
    for idx in range(10):
        sgm_2.add_encoding(enc_type=str(idx), value=300, encoder=n_encoder)

    start_time = time.time()
    for i in range(1000):
        d_4 = sgm_1.similarity(sgm=sgm_2)
    end_time = time.time()
    print((end_time - start_time)/1000)

    for val in range(110, 310, 10):
        sgm_t = SGM()
        for idx in range(10):
            sgm_t.add_encoding(enc_type=str(idx), value=val, encoder=n_encoder)
        sgm_1.learn(sgm_t, learn_rate=0.7)

    start_time = time.time()
    for i in range(1000):
        d_4 = sgm_1.similarity(sgm=sgm_2)
    end_time = time.time()
    print((end_time - start_time) / 1000)

    s_encoder = StringEncoder(n_bits=40,
                              enc_size=2048,
                              seed=12345)

    sgm_3 = SGM()
    for val in range(0, 10):
        sgm_3.add_encoding(enc_type=str(val), value=str(val), encoder=s_encoder)

    sgm_4 = SGM()
    for val in range(0, 10):
        sgm_4.add_encoding(enc_type=str(val), value=str(val*100), encoder=s_encoder)

    start_time = time.time()
    for i in range(1000):
        d_4 = sgm_3.similarity(sgm=sgm_4)
    end_time = time.time()
    print((end_time - start_time) / 1000)

    for val in range(110, 310, 10):
        sgm_t = SGM()
        for idx in range(10):
            sgm_t.add_encoding(enc_type=str(idx), value=str(val), encoder=s_encoder)
        sgm_3.learn(sgm_t, learn_rate=0.7)


    start_time = time.time()
    for i in range(1000):
        d_4 = sgm_3.similarity(sgm=sgm_4)
    end_time = time.time()
    print((end_time - start_time) / 1000)


    print('finished')
