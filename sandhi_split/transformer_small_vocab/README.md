# Small Transformer for sandhi splitting
This directory contains scripts and models obtained using a small
transformer model [paper](https://arxiv.org/abs/1706.03762) supplied 
as part of the [OpenNMT-tf library](http://opennmt.net/OpenNMT-tf).

The training data is tokenized using 
[sentencepiece](https://github.com/google/sentencepiece) with a 
vocabulary of 8000 words and used to train a transformer model 
for 10000 steps with the LazyAdamOptimizer.
Most of the settings are borrowed from the example provided in OpenNMT-tf.

Training is fairly fast if you have a GPU. It took less 
than an hour to train the model for 10000 steps using a Google
Compute Engine instance with one unit of a Nvidia K80 GPU.

## Directory structure
* `transformer_small.py` defines the model. 
* `settings.yml` defines the settings for training and evaluating the model.
* `data_sources.yml` defines the data for training and evaluation.
* `nmt_client.py` is a client script for interacting with the tensorflow server running the final exported model (in export subdirectory). It currently only supports batch mode and the detokenization should be done manually (See usage below).
* subdirectory `data` contains the data for training and testing.
* subdirectory `export` contains an exported model that can be used with tensorflow serving. Note that if you run the training yourself, the resulting directory structure will be different (see usage below).

## Training

### Pre-requisites
Install `OpenNMT-tf` and `sentencepiece` before running the scripts. 
The exported model can be served using `tensorflow_serving`.

### Data preparation
The `data` subdirectory already contains the required models 
and tokenized data. Follow the steps below if you want to 
regenerate them or modify the vocabulary size.

Set the vocabulary size parameter in `prepare_data.sh`and execute the script.
This will train a `sentencepiece` model on the training data,
tokenize the data and extract the vocabulary for the NMT model.

### Train the NMT model

# LICENSE
The code and models in this project are made available under the MIT license. The training and testing data is borrowed from [this project](https://github.com/cvikasreddy/skt/)
