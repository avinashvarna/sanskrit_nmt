# Small Transformer for sandhi splitting
This directory contains scripts and models obtained using a small
[transformer model](https://arxiv.org/abs/1706.03762) supplied 
as part of the [OpenNMT-tf library](http://opennmt.net/OpenNMT-tf).

The training data is tokenized using 
[sentencepiece](https://github.com/google/sentencepiece) with a 
vocabulary of 8000 words and used to train a transformer model 
for 10000 steps with the LazyAdamOptimizer.
Most of the settings are borrowed from the example provided in OpenNMT-tf.

Training is fairly fast if you have a GPU. It took less 
than an hour to train the model for 10000 steps using a Google
Compute Engine instance with one unit of an Nvidia K80 GPU.

## Directory structure
* `transformer_small.py` defines the model. 
* `settings.yml` defines the settings for training and evaluating the model.
* `data_sources.yml` defines the data for training and evaluation.
* `nmt_client.py` is a client script for interacting with the tensorflow server running 
the final exported model (in export subdirectory). It currently only supports batch mode 
and the detokenization should be done manually (See usage below).
* subdirectory `data` contains the data for training and testing.
* subdirectory `export` contains an exported model that can be used with tensorflow serving. 
Note that if you run the training yourself, the resulting directory structure will be different (see usage below).

## Usage

If you just want to use the provided model, skip straight to the "Serve the exported model" section below.

### Pre-requisites
Install `OpenNMT-tf` and `sentencepiece` before running the scripts. 
The exported model can be served using `tensorflow_serving`.

### Data preparation
The `data` subdirectory already contains the required models 
and tokenized data. Follow the steps below if you want to 
regenerate them or modify the vocabulary size.

* Set the `vocab_size` parameter in `prepare_data.sh`to the desired setting.
* Execute the script. This will train a `sentencepiece` model on the training data,
tokenize the data and extract the vocabulary for the NMT model.

### Train the NMT model
* Modify the training configuration in `settings.yml` as desired. E.g. to change the evaluation interval, 
the number of checkpoints to retain, etc.
* Execute `./scripts/train_and_eval.sh`. This should start training the model and log events and 
save checkpoints in `model_dir`
* If you want to monitor the training progress, run `tensorboard --logdir model_dir --port <port>`
and open the webpage in your browser. You can set port to an available port.
* OpenNMT-tf periodically exports the model under `model_dir/export/latest/`. See OpenNMT-tf documentation.

### Average the checkpoints
Run `./scripts/average_checkpoints.sh` to average the last 5 checkpoints. This typically improves the model accuracy.
Adjust the number of checkpoints to average as desired.

### Evaluate the averaged checkpoint
Run `./scripts/eval_checkpoint.sh <path_to_checkpoint> > <output_file>` to evaluate the averaged checkpoint. 
E.g. `./scripts/eval_checkpoint.sh model_dir/avg_10000 > data/avg_10000_sp.txt` if you ran the `average_checkpoints.sh` script 
without modification. This will save the output of the model in `data/avg_10000_sp.txt` which can be used to evaluate the split accuracy (see below).

### Export the averaged checkpoint for serving
Run `./scripts/export_checkpoint.sh <path_to_checkpoint>` to export the averaged checkpoint. 
This exported model is much smaller in size as it does not contain parts of the graph only required for training.
The exported model can be deployed using `tensorflow_serving`. Note that OpenNMT-tf will export the model under
`model_dir/export/manual/<timestamp>`. This has been moved to the `export` directory in this repo.

### Serve the exported model using tensorflow server
* Adjust the batching parameters in `scripts/batching_parameters.txt` as necessary. 
* Set the path to the model in `scripts/start_tf_server.sh` if you are using an exported 
model you trained yourself. The script uses the model in this repo by default.
* Run `./scripts/start_tf_server.sh` to serve the model. 
You should see something like the following if the server started successfully:
```
2018-06-12 05:59:12.764836: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: sandhi_split version: 1}
2018-06-12 05:59:12.765833: I tensorflow_serving/model_servers/main.cc:323] Running ModelServer at 0.0.0.0:9000 ...
```

### Run the client with the test data
* `./scripts/run_client.sh data/test.txt data/serving_test.txt` will run the client on the first 10 sentences in the test data and store the tokenized output in `data/serving_test.txt`
* To evaluate all the test data, run `./scripts/run_client.sh data/input_test.txt data/serving_10000.txt`


### Evaluate the split accuracy
For example with [sacrebleu](https://raw.githubusercontent.com/awslabs/sockeye/master/contrib/sacrebleu/sacrebleu.py):
```
$$ ./sacrebleu.py data/output_test.txt -m bleu chrf < data/serving_10000.txt
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.2.3 = 83.74 92.9/87.3/82.4/78.3 (BP = 0.985 ratio = 0.985 hyp_len = 32136 ref_len = 32638)
chrF2+case.mixed+numchars.6+numrefs.1+space.False+tok.13a+version.1.2.3 = 0.96
```
showing that the model output obtained a BLEU score of 83.74 and chrF = 0.96 with respect to the reference splits. 
Note that these may not be the best metrics to use for this task, but are commonly used in NMT.

### Interactively split a sentence
`scripts/sandhi_split.sh` can be used to split a single input sentence on the command line. E.g.
```
./scripts/sandhi_split.sh "yasmAdAha"
Input: yasmAdAha
Split: yasmAt Aha
```

## TODO
* Explore if quantizing/pruning the weights can further reduce the exported model size.

# LICENSE
The code and models in this project are made available under the MIT license. The training and testing data is borrowed from [this project](https://github.com/cvikasreddy/skt/)
