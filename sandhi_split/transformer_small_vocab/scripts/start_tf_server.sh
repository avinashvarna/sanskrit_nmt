#!/bin/bash

tensorflow_model_server --port=9000 --model_name=sandhi_split --model_base_path=${PWD}/export --enable_batching=true --batching_parameters_file=scripts/batching_parameters.txt
