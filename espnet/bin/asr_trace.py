#!/usr/bin/env python3
# encoding: utf-8
"""End-to-end speech recognition model tracing script."""

import os
import sys

import configargparse
import torch
import torchaudio

from espnet.asr.pytorch_backend.trace import TraceModel


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description='Trace model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_path', type=str, default='exp/acoustic/model.loss.best',
                        help='The acoustic model path')
    parser.add_argument('--lm_path', type=str, default='exp/language/rnnlm.model.best',
                        help='The language model path')
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    tm = TraceModel(model_path=args.model_path,
                    lm_path=args.lm_path)

    waveform, sample_rate = torchaudio.load_wav('wav/arabic.wav')
    module = torch.jit.trace(tm, waveform, check_trace=False)
    module.save(os.path.join(os.path.dirname(args.model_path), '{}_traced.pt'.format(os.path.basename(args.model_path))))

    # Smoke test the model
    print('Starting to serve the model: '.format(args.model_path))
    for res_tuple in module(waveform):
        print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))


if __name__ == '__main__':
    main(sys.argv[1:])
