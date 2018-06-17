"""Translation client.

Modified opennmt-tf example using some ideas from
tensorflow serving mnist example.
."""

from __future__ import print_function

import argparse

from grpc.beta import implementations

from tensorflow_serving.apis import prediction_service_pb2

import codecs
import operator
import threading
import progressbar

import sentencepiece as spm

from client_common import translate, parse_translation_result


# List to hold the translation results
results = []


class _RateLimiter(object):
    """Rate limiter to control number of concurrent requests."""

    def __init__(self, concurrency):
        self._concurrency = concurrency
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

    def wait_done(self, i):
        with self._condition:
            while self._done != i:
                self._condition.wait()


def _create_rpc_callback(i, rate_limiter, sentence_processor):
    """Creates RPC callback function.
    Args:
        i: index of query
        rate_limiter: A `_RateLimiter` object
    Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.
        Args:
            result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            print("Exception occurred for ", i)
            print(exception)
        else:
            response = parse_translation_result(result_future.result(),
                                                sentence_processor)
            results.append((i, response))
        rate_limiter.inc_done()
        rate_limiter.dec_active()
    return _callback


def main():
    parser = argparse.ArgumentParser(description="Translation client example")
    parser.add_argument("--model_name", required=True,
                        help="model name")
    parser.add_argument("--host", default="localhost",
                        help="model server host")
    parser.add_argument("--port", type=int, default=9000,
                        help="model server port")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="request timeout")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="number of concurrent requests")
    parser.add_argument('--spm_model', type=str,
                        help="sentencepiece model file")
    parser.add_argument('--input_file', type=str, help="input file")
    parser.add_argument('--output_file', type=str, help="output file")
    args = parser.parse_args()

    channel = implementations.insecure_channel(args.host, args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    rate_limiter = _RateLimiter(args.concurrency)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    bar = progressbar.ProgressBar()
    with codecs.open(args.input_file, "r", "utf8") as f:
        for i, line in bar(enumerate(f)):

            rate_limiter.throttle()
            future = translate(stub, args.model_name, sp, line.strip(),
                               timeout=args.timeout)
            future.add_done_callback(_create_rpc_callback(i, rate_limiter, sp))

    rate_limiter.wait_done(i+1)
    results.sort(key=operator.itemgetter(0))

    with codecs.open(args.output_file, "w", "utf8") as f:
        for r in results:
            f.write(r[1] + "\n")


if __name__ == "__main__":
    main()
