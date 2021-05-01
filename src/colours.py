#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from random import choices, choice, randint, random
import csv


def make_colours():

    clients = ['ABC', 'DEF', 'GHI']
    seq = ['RED', 'ORANGE', 'YELLOW', 'GREEN', 'BLUE']
    anomalies = ['PURPLE', 'BROWN', 'GREY', 'BLACK', 'TURQUOISE']

    colours = {'RED': {'r': 255, 'g': 0, 'b': 0},
               'ORANGE': {'r': 255, 'g': 129, 'b': 0},
               'YELLOW': {'r': 255, 'g': 233, 'b': 0},
               'GREEN': {'r': 0, 'g': 202, 'b': 14},
               'BLUE': {'r': 22, 'g': 93, 'b': 239},
               'PURPLE': {'r': 166, 'g': 1, 'b': 214},
               'BROWN': {'r': 151, 'g': 76, 'b': 2},
               'GREY': {'r': 128, 'g': 128, 'b': 128},
               'BLACK': {'r': 0, 'g': 0, 'b': 0},
               'TURQUOISE': {'r': 150, 'g': 255, 'b': 255},
               }

    noise = 40
    rows = []
    row_id = 0
    for order_id in range(100):
        client = choice(clients)
        for i in range(randint(1, len(seq))):
            row = {'Row_id': row_id,
                   'Client': client,
                   'Order_id': order_id,
                   'Product': seq[i],
                   'RGB_Red': colours[seq[i]]['r'] + int(random() * noise) if colours[seq[i]]['r'] < (255 - noise) else colours[seq[i]]['r'] - int(random() * noise),
                   'RGB_Green': colours[seq[i]]['g'] + int(random() * noise) if colours[seq[i]]['g'] < (255 - noise) else colours[seq[i]]['g'] - int(random() * noise),
                   'RGB_Blue': colours[seq[i]]['b'] + int(random() * noise) if colours[seq[i]]['b'] < (255 - noise) else colours[seq[i]]['b'] - int(random() * noise)}
            rows.append(row)
            row_id += 1

    with open('training_40.csv', 'w', newline='') as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    noise = 40
    rows = []
    row_id = 0
    for order_id in range(10):
        client = choice(clients)
        if random() < 0.9:
            # normal data
            for i in range(randint(1, len(seq))):
                row = {'Row_id': row_id,
                       'Client': client,
                       'Order_id': order_id,
                       'Product': seq[i],
                       'RGB_Red': colours[seq[i]]['r'] + int(random() * noise) if colours[seq[i]]['r'] < (255 - noise) else colours[seq[i]]['r'] - int(random() * noise),
                       'RGB_Green': colours[seq[i]]['g'] + int(random() * noise) if colours[seq[i]]['g'] < (255 - noise) else colours[seq[i]]['g'] - int(random() * noise),
                       'RGB_Blue': colours[seq[i]]['b'] + int(random() * noise) if colours[seq[i]]['b'] < (255 - noise) else colours[seq[i]]['b'] - int(random() * noise)}
                rows.append(row)
                row_id += 1
        else:
            # abnormal data
            #
            print('abnormal order:', order_id)
            for i in range(randint(1, len(anomalies))):
                row = {'Row_id': row_id,
                       'Client': client,
                       'Order_id': order_id,
                       'Product': anomalies[i],
                       'RGB_Red': colours[anomalies[i]]['r'] + int(random() * noise) if colours[anomalies[i]]['r'] < (255 - noise) else colours[anomalies[i]]['r'] - int(random() * noise),
                       'RGB_Green': colours[anomalies[i]]['g'] + int(random() * noise) if colours[anomalies[i]]['g'] < (255 - noise) else colours[anomalies[i]]['g'] - int(random() * noise),
                       'RGB_Blue': colours[anomalies[i]]['b'] + int(random() * noise) if colours[anomalies[i]]['b'] < (255 - noise) else colours[anomalies[i]]['b'] - int(random() * noise)}
                rows.append(row)
                row_id += 1

    with open('test_40.csv', 'w', newline='') as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('finished')


if __name__ == '__main__':
    make_colours()