import os

import torch
import torchaudio

from espnet.asr.pytorch_backend.trace import TraceModel

model_path = 'exp/acoustic/model.loss.best'
lm_path = 'exp/language/rnnlm.model.best'
tm = TraceModel(model_path=model_path,
                lm_path=lm_path)
waveform, sample_rate = torchaudio.load_wav('wav/arabic.wav')
module = torch.jit.trace(tm, waveform, check_trace=False)
module.save(os.path.join(os.path.dirname(model_path), '{}_traced.pt'.format(os.path.basename(model_path))))

# Smoke test the model
for res_tuple in module(waveform):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))