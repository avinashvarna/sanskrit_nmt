"""Translation client.

Modified opennmt-tf example using some ideas from
tensorflow serving mnist example.
."""

from __future__ import print_function

import argparse

from grpc.beta import implementations

from tensorflow_serving.apis import prediction_service_pb2

import sentencepiece as spm

from client_common import translate, parse_translation_result


# List to hold the translation results
results = []


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
    parser.add_argument('input', type=str, help="string to split")
    args = parser.parse_args()

    channel = implementations.insecure_channel(args.host, args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    future = translate(stub, args.model_name, sp, args.input,
                       timeout=args.timeout)
    output = parse_translation_result(future.result(), sp)
    print("Input:", args.input)
    print("Split:", output)


if __name__ == "__main__":
    main()
