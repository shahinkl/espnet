from argparse import Namespace

import torch
import torchaudio

from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
import logging
from espnet.nets.lm_interface import dynamic_import_lm


class TraceModel(torch.nn.Module):
    def __init__(self, model_path: str,
                 lm_path: str):
        super(TraceModel, self).__init__()
        self.model, self.train_args = load_trained_model(model_path=model_path)
        logging.info(self.model)
        assert isinstance(self.model, ASRInterface)
        self.model.eval()
        self.recog_args = self.__get_recog_args()
        self.rnnlm = self.__make_lm_module(lm_path=lm_path)

    def get_sentence(self, labels):
        return "".join([self.train_args.char_list[char_index] for char_index in list(labels.numpy())])

    def forward(self, audio_data):
        fbank = torchaudio.compliance.kaldi.fbank(audio_data, num_mel_bins=80, window_type='povey')
        decoded_list = self.model.recognize(fbank, self.recog_args, char_list=self.train_args.char_list, rnnlm=self.rnnlm)
        result = []
        for decoded in decoded_list:
            result.append((torch.tensor(decoded['score'], dtype=torch.float32), torch.IntTensor(decoded['yseq'])))
        return tuple(result)

    @staticmethod
    def __get_recog_args():
        recog_args = Namespace()
        recog_args.ctc_weight = 0.4
        recog_args.beam_size = 6
        recog_args.penalty = 0.0
        recog_args.maxlenratio = 0.0
        recog_args.minlenratio = 0.0
        recog_args.lm_weight = 0.0
        recog_args.nbest = 10
        return recog_args

    def __make_lm_module(self, lm_path: str):
        if len(lm_path) <= 0:
            return None
        lm_config = get_model_conf(model_path=lm_path)
        lm_model_module = getattr(lm_config, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_config.backend)
        lm = lm_class(len(self.train_args.char_list), lm_config)
        torch_load(lm_path, lm)
        lm.eval()
        return lm
