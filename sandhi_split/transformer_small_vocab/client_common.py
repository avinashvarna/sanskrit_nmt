"""Translation client common functions

Modified opennmt-tf example using some ideas from
tensorflow serving mnist example.
."""

from __future__ import print_function
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2

import codecs


def parse_translation_result(result, sentence_processor):
    """Parses a translation result.

    Args:
        result: A `PredictResponse` proto.
        sentence_processor: A `sentencepiece.SentenceProcessor`

    Returns:
        A result string
    """
    lengths = tf.make_ndarray(result.outputs["length"])[0]
    hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]

    # Only consider the first hypothesis (the best one).
    best_hypothesis = hypotheses[0]
    best_length = lengths[0]
    model_out = best_hypothesis[0:best_length - 1]    # Ignore </s>
    pieces = sentence_processor.DecodePieces(list(model_out))
    return codecs.decode(pieces)


def translate(stub, model_name, sentence_processor,
              input_string, timeout=5.0):
    """Translates a sequence of tokens.

    Args:
        stub: The prediction service stub.
        model_name: The model to request.
        sentence_processor: A `sentencepience.SentenceProcessor`
        input_string: The input string
        timeout: Timeout after this many seconds.

    Returns:
        A future.
    """
    tokens = sentence_processor.EncodeAsPieces(input_string)
    length = len(tokens)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
            tf.make_tensor_proto([tokens], shape=(1, length)))
    request.inputs["length"].CopyFrom(
            tf.make_tensor_proto([length], shape=(1,)))

    return stub.Predict.future(request, timeout)
