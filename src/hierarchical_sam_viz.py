#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import plotly.graph_objects as go


def plot_sam(sam, raw_data, xyz_types, colour_nodes, norm_colour=False):

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

    nos_mapped_data = 0

    neuron_xyz = {}
    for neuron_key in sam['neurons']:
        nos_mapped_data += sam['neurons'][neuron_key]['n_bmu']
        for enc_type in sam['neurons'][neuron_key]['sgm']['encodings']:
            if enc_type == xyz_types[0]:
                node_x.append(sam['neurons'][neuron_key]['sgm']['encodings'][enc_type])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'x': sam['neurons'][neuron_key]['sgm']['encodings'][enc_type], 'z': 0.0, 'y': 0.0}
                else:
                    neuron_xyz[neuron_key]['x'] = sam['neurons'][neuron_key]['sgm']['encodings'][enc_type]
            elif enc_type == xyz_types[1]:
                node_y.append(sam['neurons'][neuron_key]['sgm']['encodings'][enc_type])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'y': sam['neurons'][neuron_key]['sgm']['encodings'][enc_type], 'x': 0.0, 'z': 0.0}
                else:
                    neuron_xyz[neuron_key]['y'] = sam['neurons'][neuron_key]['sgm']['encodings'][enc_type]

            elif len(xyz_types) == 3 and enc_type == xyz_types[2]:

                node_z.append(sam['neurons'][neuron_key]['sgm']['encodings'][enc_type])
                if neuron_key not in neuron_xyz:
                    neuron_xyz[neuron_key] = {'z': sam['neurons'][neuron_key]['sgm']['encodings'][enc_type], 'x': 0.0, 'y': 0.0}
                else:
                    neuron_xyz[neuron_key]['z'] = sam['neurons'][neuron_key]['sgm']['encodings'][enc_type]

            if colour_nodes is not None and enc_type == colour_nodes:
                if enc_type not in colour_labels:
                    colour = next_color
                    colour_labels[enc_type] = next_color
                    next_color += 1
                else:
                    colour = colour_labels[enc_type]
                node_colour.append(colour)

        if len(xyz_types) == 2:
            node_z.append(0.0)

        if colour_nodes is None and norm_colour:

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
        node_size.append(10 + sam['neurons'][neuron_key]['n_bmu'])

    pairs = set()
    for neuron_key in sam['neurons']:
        for nn_key in sam['neurons'][neuron_key]['nn']:
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

    if colour_nodes is None:
        for idx in range(len(node_x)):
            if norm_colour:
                r = max(min(int(255 * ((node_x[idx] - x_min_max['min']) / (x_min_max['max'] - x_min_max['min']))), 255), 0)
                g = max(min(int(255 * ((node_y[idx] - y_min_max['min']) / (y_min_max['max'] - y_min_max['min']))), 255), 0)
                b = max(min(int(255 * ((node_z[idx] - z_min_max['min']) / (z_min_max['max'] - z_min_max['min']))), 255), 0)
            else:
                r = int(node_x[idx])
                g = int(node_y[idx])
                b = int(node_z[idx])

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

    if nos_mapped_data > 0:
        compression_ratio = round(len(sam["neurons"]) / nos_mapped_data, 2)
    else:
        compression_ratio = 1.0

    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{sam["name"]} Nos Neurons: {len(sam["neurons"])} Nos Raw Data:{nos_mapped_data} Ratio: {compression_ratio}'))
    fig.show()


def plot_pors(pors, name=None):
    anomaly_threshold = []
    motif_threshold = []
    anomalies = []
    motifs = []
    ema_similarity = []
    ema_stdev = []

    x_values = []
    similarity = []
    nos_neurons = []

    nos_mapped_data = 0

    sam_name = None

    final_neurons = 0
    for x in range(len(pors)):

        x_values.append(x)
        por = None
        if name is not None:
            if name in pors[x]:
                por = pors[x][name]
        else:
            por = pors[x]

        if por is not None:
            final_neurons = por['nos_neurons']
            sam_name = por['sam']
            nos_mapped_data += 1
            similarity.append(por['bmu_similarity'])
            nos_neurons.append(por['nos_neurons'])
            anomaly_threshold.append(por['anomaly_threshold'])
            motif_threshold.append(por['motif_threshold'])
            ema_similarity.append(por['ema_similarity'])
            ema_stdev.append(pow(por['ema_variance'], 0.5))

            if por['anomaly']:
                anomalies.append(por['bmu_similarity'])
            else:
                anomalies.append(None)
            if por['motif']:
                motifs.append(por['bmu_similarity'])
            else:
                motifs.append(None)
        else:
            if len(similarity) > 0:
                similarity.append(similarity[-1])
            else:
                similarity.append(None)

            if len(nos_neurons) > 0:
                nos_neurons.append(nos_neurons[-1])
            else:
                nos_neurons.append(None)

            if len(anomaly_threshold) > 0:
                anomaly_threshold.append(anomaly_threshold[-1])
            else:
                anomaly_threshold.append(None)

            if len(motif_threshold) > 0:
                motif_threshold.append(motif_threshold[-1])
            else:
                motif_threshold.append(None)

            if len(ema_similarity) > 0:
                ema_similarity.append(ema_similarity[-1])
            else:
                ema_similarity.append(None)

            if len(ema_stdev) > 0:
                ema_stdev.append(ema_stdev[-1])
            else:
                ema_stdev.append(None)

            anomalies.append(None)
            motifs.append(None)

    if final_neurons > 0:
        nos_neurons = [(nos_neurons[idx] / final_neurons) if nos_neurons[idx] is not None else None for idx in range(len(nos_neurons))]

    similarity_scatter = go.Scatter(x=x_values, y=similarity, mode='lines', name='similarity', line=dict(width=2, color='black'))
    nos_neurons_scatter = go.Scatter(x=x_values, y=nos_neurons, mode='lines', name='nos neurons', line=dict(width=2, color='orange'))

    anomaly_threshold_scatter = go.Scatter(x=x_values, y=anomaly_threshold, mode='lines', name='anomaly threshold', line=dict(width=2, color='red'))
    motif_threshold_scatter = go.Scatter(x=x_values, y=motif_threshold, mode='lines', name='motif threshold', line=dict(width=2, color='green'))
    ema_similarity_scatter = go.Scatter(x=x_values, y=ema_similarity, mode='lines', name='ema similarity', line=dict(width=2, color='blue'))
    ema_stdev_scatter = go.Scatter(x=x_values, y=ema_stdev, mode='lines', name='stdev similarity', line=dict(width=2, color='purple'))

    anomalies_scatter = go.Scatter(x=x_values, y=anomalies, mode='markers', name='anomaly', marker=dict(size=10, color='red', opacity=1.0, symbol='square'))
    motifs_scatter = go.Scatter(x=x_values, y=motifs, mode='markers', name='motif', marker=dict(size=10, color='green', opacity=1.0, symbol='square'))

    fig = go.Figure(data=[nos_neurons_scatter, similarity_scatter, anomaly_threshold_scatter, motif_threshold_scatter, ema_similarity_scatter,ema_stdev_scatter, anomalies_scatter, motifs_scatter])

    compression_ratio = round(final_neurons / nos_mapped_data, 2)
    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{sam_name} nos neurons: {final_neurons} Nos Raw_data: {nos_mapped_data} Ratio: {compression_ratio}'))
    fig.show()
