import os

import torch
import torchaudio
from kaldi.matrix import Vector
from kaldi.feat.pitch import compute_and_process_kaldi_pitch
from kaldi.feat.pitch import ProcessPitchOptions
from kaldi.feat.pitch import PitchExtractionOptions
from kaldi.feat.fbank import Fbank, FbankOptions
import numpy
import kaldi_io
import kaldiio
from scipy.io import wavfile

from espnet.asr.pytorch_backend.trace import TraceModel

cmvn_file = '/data/all_data/train_960/cmvn.ark'


def compute_pitch_feats_and_post(data):
    pitch_opts = PitchExtractionOptions()
    post_opts = ProcessPitchOptions()
    wav_vector = Vector(data)
    feats = compute_and_process_kaldi_pitch(pitch_opts, post_opts, wav_vector)
    feats_data = feats.numpy()
    return feats_data


def fbank_feats(wave_numpy, wave_tensor):
    feats_pitch = compute_pitch_feats_and_post(wave_numpy)
    f_bank = torchaudio.compliance.kaldi.fbank(
        wave_tensor, blackman_coeff=0.42, channel=-1, dither=0.0, energy_floor=0.0, frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=True, low_freq=20.0,
        min_duration=0.0, num_mel_bins=80)
    fbank_pitch = numpy.concatenate([f_bank, feats_pitch], axis=1)
    return fbank_pitch
    # return torch.tensor(fbank_pitch, dtype=torch.float32)


model_path = 'exp/transformer_960/results/model.val5.avg.best'
lm_path = 'exp/train_custom_transformer_pytorch_lm_unigram5000_ngpu4/rnnlm.model.best'
tm = TraceModel(model_path=model_path,
                lm_path=lm_path)
waveform, sample_rate = torchaudio.load_wav('wavs/1589065646228.wav')
print('waveform: {}'.format(waveform))
sample_rate2, wave_npy = wavfile.read('wavs/1589065646228.wav')
print('wave_npy: {}'.format(wave_npy))
fbank_pitch = fbank_feats(wave_npy, waveform)
norm = torch.load("cmvn.bin")
count = norm[0][-1]
mean = norm[0][:-1] / count
var = (norm[1][:-1] / count) - mean * mean
scale = 1.0 / numpy.sqrt(var)
offset = - (mean * scale)

speech_in_features_normalized = fbank_pitch * scale + offset
speech_in_features_normalized = torch.as_tensor(speech_in_features_normalized)
print('fbank: {}'.format(speech_in_features_normalized))
fbank = torch.as_tensor(fbank_pitch)
module = torch.jit.trace(tm, fbank, check_trace=False)
# module.save(os.path.join(os.path.dirname(model_path), '{}_traced.pt'.format(os.path.basename(model_path))))

# Smoke test the model
for res_tuple in module(fbank):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))
