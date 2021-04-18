#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import plotly.graph_objects as go


def plot_sam(sam, raw_data, xyz_types, colour_nodes, temporal_key=0):

    node_x = []
    node_y = []
    node_z = []

    next_color = 10
    colour_labels = {}
    node_colour = []
    node_label = []
    node_size = []

    edge_x = []
    edge_y = []
    edge_z = []

    x_min_max = {'min': None, 'max': None}
    y_min_max = {'min': None, 'max': None}
    z_min_max = {'min': None, 'max': None}

    neuron_xyz = {}
    neurons = sam['neurons']['neuron_to_bit']
    for neuron_key in neurons:
        for enc_key in neurons[neuron_key]['sgm']['encoding'][temporal_key]:
            if enc_key == xyz_types[0]:
                node_x.append(neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'x': neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key], 'z': 0.0, 'y': 0.0}
                else:
                    neuron_xyz[neuron_key]['x'] = neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key]
            elif enc_key == xyz_types[1]:
                node_y.append(neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'y': neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key], 'x': 0.0, 'z': 0.0}
                else:
                    neuron_xyz[neuron_key]['y'] = neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key]

            elif len(xyz_types) == 3 and enc_key == xyz_types[2]:

                node_z.append(neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'z': neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key], 'x': 0.0, 'y': 0.0}
                else:
                    neuron_xyz[neuron_key]['z'] = neurons[neuron_key]['sgm']['encoding'][temporal_key][enc_key]

            if colour_nodes is not None and enc_key == colour_nodes:
                if enc_key not in colour_labels:
                    colour = next_color
                    colour_labels[enc_key] = next_color
                    next_color += 1
                else:
                    colour = colour_labels[enc_key]
                node_colour.append(colour)

        if len(xyz_types) == 2:
            node_z.append(0.0)

        if colour_nodes is None:

            if x_min_max['max'] is None or node_x[-1] > x_min_max['max']:
                x_min_max['max'] = node_x[-1]
            if x_min_max['min'] is None or node_x[-1] < x_min_max['min']:
                x_min_max['min'] = node_x[-1]
            if y_min_max['max'] is None or node_y[-1] > y_min_max['max']:
                y_min_max['max'] = node_y[-1]
            if y_min_max['min'] is None or node_y[-1] < y_min_max['min']:
                y_min_max['min'] = node_y[-1]
            if z_min_max['max'] is None or node_z[-1] > z_min_max['max']:
                z_min_max['max'] = node_z[-1]
            if z_min_max['min'] is None or node_z[-1] < z_min_max['min']:
                z_min_max['min'] = node_z[-1]

        node_label.append(neuron_key)
        node_size.append(10 + neurons[neuron_key]['n_bmu'])

    """
    pairs = set()
    for neuron_key in neurons:
        for nn_key in neurons[neuron_key]['nn']:
            pair = (min(neuron_key, nn_key), max(neuron_key, nn_key))
            if pair not in pairs:
                pairs.add(pair)
                edge_x.append(neuron_xyz[neuron_key]['x'])
                edge_x.append(neuron_xyz[nn_key]['x'])
                edge_x.append(None)

                edge_y.append(neuron_xyz[neuron_key]['y'])
                edge_y.append(neuron_xyz[nn_key]['y'])
                edge_y.append(None)

                if len(xyz_types) == 3:
                    edge_z.append(neuron_xyz[neuron_key]['z'])
                    edge_z.append(neuron_xyz[nn_key]['z'])
                else:
                    edge_z.append(0.0)
                    edge_z.append(0.0)

                edge_z.append(None)
    """

    if colour_nodes is None:
        for idx in range(len(node_x)):
            r = max(min(int(255 * ((node_x[idx] - x_min_max['min']) / (x_min_max['max'] - x_min_max['min']))), 255), 0)
            g = max(min(int(255 * ((node_y[idx] - y_min_max['min']) / (y_min_max['max'] - y_min_max['min']))), 255), 0)
            if (z_min_max['max'] - z_min_max['min']) > 0:
                b = max(min(int(255 * ((node_z[idx] - z_min_max['min']) / (z_min_max['max'] - z_min_max['min']))), 255), 0)
            else:
                b = 0
            node_colour.append(f'rgb({r},{g},{b})')

    raw_x = []
    raw_y = []
    raw_z = []
    raw_size = []
    raw_colour = []
    for idx in range(len(raw_data)):
        raw_x.append(raw_data[idx][0])
        raw_y.append(raw_data[idx][1])
        if len(raw_data[idx]) == 3:
            raw_z.append(raw_data[idx][2])
        else:
            raw_z.append(0.0)
        raw_size.append(5)
        raw_colour.append(1)

    raw_scatter = go.Scatter3d(x=raw_x, y=raw_y, z=raw_z, mode='markers',  name='raw data', marker=dict(size=3, color=raw_colour, opacity=1.0, symbol='square'))
    neuron_scatter = go.Scatter3d(x=node_x, y=node_y, z=node_z, name='neuron', hovertext=node_label, mode='markers+text', marker=dict(size=node_size, color=node_colour, opacity=0.7))
    edge_scatter = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, name='nn edge', mode='lines', line=dict(width=1, color='grey'))

    fig = go.Figure(data=[raw_scatter, edge_scatter, neuron_scatter])

    compression_ratio = round(len(neurons) / len(raw_data), 2)

    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{sam["name"]} Nos Neurons: {len(neurons)} Nos Raw Data:{len(raw_data)} Ratio: {compression_ratio}'))
    fig.show()


def plot_pors(pors):
    ema_similarity = []
    ema_stdev = []

    x_values = []
    similarity = []
    nos_neurons = []

    final_neurons = pors[-1]['nos_neurons']

    for x in range(len(pors)):

        x_values.append(x)
        por = pors[x]

        similarity.append(por['bmu_similarity'])
        nos_neurons.append(por['nos_neurons'] / final_neurons)
        ema_similarity.append(por['avg_bmu_similarity'])
        ema_stdev.append(por['std_bmu_similarity'])

    anomaly_score_scatter = go.Scatter(x=x_values, y=similarity, mode='lines', name='similarity', line=dict(width=2, color='black'))
    nos_neurons_scatter = go.Scatter(x=x_values, y=nos_neurons, mode='lines', name='nos neurons', line=dict(width=2, color='orange'))

    ema_similarity_scatter = go.Scatter(x=x_values, y=ema_similarity, mode='lines', name='avg similarity', line=dict(width=2, color='blue'))
    ema_stdev_scatter = go.Scatter(x=x_values, y=ema_stdev, mode='lines', name='std similarity', line=dict(width=2, color='purple'))

    fig = go.Figure(data=[nos_neurons_scatter, anomaly_score_scatter, ema_similarity_scatter,ema_stdev_scatter])

    por = pors[-1]

    compression_ratio = round(final_neurons / len(pors), 2)
    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{por["name"]} nos neurons: {final_neurons} Nos Raw_data: {len(pors)} Ratio: {compression_ratio}'))
    fig.show()
