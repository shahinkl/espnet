from argparse import Namespace

import torch
import torchaudio

from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import BeamSearch
import logging


class TraceModel(torch.nn.Module):
    def __init__(self, model_path: str):
        super(TraceModel, self).__init__()
        self.model, self.train_args = load_trained_model(model_path=model_path)
        logging.info(self.model)
        assert isinstance(self.model, ASRInterface)
        self.model.eval()
        self.recog_args = self.__get_recog_args()

        # self.bs = BeamSearch(scorers=self.model.scorers(),
        #                      weights={"decoder": 0.6, "ctc": 0.4},
        #                      sos=self.model.sos,
        #                      eos=self.model.eos,
        #                      beam_size=1,
        #                      vocab_size=len(self.train_args.char_list))

    def get_sentence(self, labels):
        return "".join([self.train_args.char_list[char_index] for char_index in list(labels.numpy())])

    def compute_pitch_feats_and_post(self, data):
        pitch_opts = PitchExtractionOptions()
        post_opts = ProcessPitchOptions()
        wav_vector = Vector(data)
        feats = compute_and_process_kaldi_pitch(pitch_opts, post_opts, wav_vector)
        feats_data = feats.numpy()
        return feats_data

    def forward(self, audio_data):
        fbank = torchaudio.compliance.kaldi.fbank(audio_data, num_mel_bins=80, window_type='povey')

        decoded_list = self.model.recognize(inputs, self.recog_args, char_list=self.train_args.char_list, use_jit=False)
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
