# sanskrit_nmt
Training NMT models for Sanskrit

Experiments with training various Neural Machine Translation (NMT) architectures for various Sanskrit NLP tasks (sandhi splitting, tagging, etc.).

## Sandhi splitting using Transformer model
[sandhi_split/transformer_small_vocab](./sandhi_split/transformer_small_vocab) has scripts + output model for sandhi splitting using a transformer model provided by the OpenNMT-tf library. A small model and vocabulary were used to ensure a small model size and fast training. The results are quite good for a small model that can be trained in < 1hr on a good GPU. See the README in that directory for more details.
