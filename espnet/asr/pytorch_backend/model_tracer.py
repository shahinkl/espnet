from argparse import Namespace

from espnet.asr.pytorch_backend.asr import load_trained_model
import torch
from torchaudio.compliance.kaldi import fbank, HAMMING
from torchaudio import load, load_wav
import os

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import BeamSearch


# from espnet.utils.deterministic_utils import set_deterministic_pytorch
# from espnet.utils.io_utils import LoadInputsAndTargets
# import json
# import logging


# def trace(args):
#     set_deterministic_pytorch(args)
#     model, train_args = load_trained_model(args.model)
#
#     load_inputs_and_targets = LoadInputsAndTargets(
#         mode='asr', load_output=False, sort_in_input_length=False,
#         preprocess_conf=train_args.preprocess_conf
#         if args.preprocess_conf is None else args.preprocess_conf,
#         preprocess_args={'train': False})
#
#     with open(args.recog_json, 'rb') as f:
#         js = json.load(f)['utts']
#
#     features = None
#     with torch.no_grad():
#         for idx, name in enumerate(js.keys(), 1):
#             logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
#             batch = [(name, js[name])]
#             features = load_inputs_and_targets(batch)[0][0]
#             break
#
#     traced_model = torch.jit.trace(model, features)
#     traced_model.save(args.traced_model)

class ModelTracer(object):
    def __init__(self) -> None:
        super().__init__()

    '''
        To trace the model for serving.
        Args:
            model_path (str): Path to model.***.best
    
    '''

    def trace_model(self, model_path: str, is_bs_decode: bool = True):

        model, train_args = load_trained_model(model_path=model_path)
        assert isinstance(model, ASRInterface)
        model.eval()
        traced_model = torch.jit.trace(model.recognize, [self.__get_sample_fbank(), self.__get_recog_args()])
        traced_model.save(os.path.join(os.path.dirname(model_path), '{}_traced.pt'.format(os.path.basename(model_path))))

    @staticmethod
    def __get_sample_fbank():
        waveform, sample_rate = load('resource/SI1657.wav')
        return fbank(waveform, num_mel_bins=83, window_type=HAMMING)

    @staticmethod
    def __get_recog_args():
        recog_args = Namespace()
        recog_args.ctc_weight = 0.5
        recog_args.beam_size = 2
        recog_args.penalty = 0.0
        recog_args.maxlenratio = 0.0
        recog_args.minlenratio = 0.0
        recog_args.lm_weight = 0.7
        recog_args.nbest = 3
        return recog_args
