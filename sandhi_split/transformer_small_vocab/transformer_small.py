"""A small transformer model."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",
          embedding_size=256),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=256),
      num_layers=3,
      num_units=256,
      num_heads=8,
      ffn_inner_dim=256,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1)
