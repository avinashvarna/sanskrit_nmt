#!/bin/bash

onmt-main train_and_eval --model transformer_small.py \
          --config settings.yml data_sources.yml
