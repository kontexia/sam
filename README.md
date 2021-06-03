# SAM

This Python library is an implementation of an explainable neural network that is based on Sparse Associative Memory (SAM)

## Sparse Associative Memory Neural Network
A SAM neural network is capable of performing the following standard machine learning functions:
* Unsupervised clustering - SAM learns generalised representations of data that are essentially clusters of similar raw data
* Supervised Classification - Include labels or classes in the data and SAM will associate all similar new data with those classes
* Temporal Prediction - SAM is capable of learning the sequences of generalisations and given a partial sequence, SAM can recall (predict) the rest of the sequence
* Anomaly Detection - SAM learns by measuring the similarity of incoming data to existing learned generalisations. A significant drop in similarity indicates the presence of an anomaly

SAM is specifically designed to learn from Sparse Distributed Representations (SDR) of data. 
SDRs encode complex numeric and categorical composite data as large sparsely populated binary vectors which have near infinite capacity. 
These encodings are capable of representing the semantic similarity between things by ensuring there is an overlap of bits in both vectors

## Encoders
SAM implements three encoders:
* Numeric - capable of encoding numeric int / float data in either linear or log scale. It ensures that numbers close together on the numeric line have a level of similarity
* Category - capable of encoding either strings with no semantic similarity or a list of semantically similar strings in a list 

## Sparse Distributed Representation

## Neural Graph

## Spatial Pooler

## SAM Fabric

