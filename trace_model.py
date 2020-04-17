# from espnet.asr.pytorch_backend.model_tracer import ModelTracer
from espnet.asr.pytorch_backend.trace import TraceModel
import torch
import torchaudio
import kaldiio
import time

fbank = kaldiio.load_mat('resource/feats.1.ark:14')
fbank_tensor = torch.FloatTensor(fbank)
# mt = ModelTracer()
# mt.trace_model(model_path='egs/librispeech/asr1/exp/train_960_pytorch_train_specaug/results/model.val5.avg.best')

tm = TraceModel(model_path='egs/librispeech/asr1/exp/train_960_pytorch_train_specaug/results/model.val5.avg.best')
waveform, sample_rate = torchaudio.load_wav('resource/2020-04-08T12_26_40.602Z.wav')
# waveform, sample_rate = torchaudio.load_wav('resource/SI1657.wav')
module = torch.jit.trace(tm, waveform, check_trace=False)
for res_tuple in module(fbank_tensor):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))

start = time.time()
for res_tuple in module(fbank_tensor):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))
end = time.time()
print(end - start)

start = time.time()
for res_tuple in module(fbank_tensor):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))
end = time.time()
print(end - start)

start = time.time()
for res_tuple in module(fbank_tensor):
    print('Score: {}, Sentence: {}\n'.format(res_tuple[0], tm.get_sentence(labels=res_tuple[1])))
end = time.time()
print(end - start)
