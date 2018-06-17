"""A small transformer model."""

from opennmt.models import Transformer
from opennmt.inputters import WordEmbedder


def model():
    source = WordEmbedder(
                          vocabulary_file_key="source_words_vocabulary",
                          embedding_size=256)
    target = WordEmbedder(
                          vocabulary_file_key="target_words_vocabulary",
                          embedding_size=256)
    return Transformer(
                       source_inputter=source,
                       target_inputter=target,
                       num_layers=3,
                       num_units=256,
                       num_heads=8,
                       ffn_inner_dim=256,
                       dropout=0.1,
                       attention_dropout=0.1,
                       relu_dropout=0.1)
