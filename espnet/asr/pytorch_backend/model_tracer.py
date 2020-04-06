from espnet.asr.pytorch_backend.asr import load_trained_model
import torch
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets
import json
import logging


def trace(args):
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    features = None
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            features = load_inputs_and_targets(batch)[0][0]
            break

    traced_model = torch.jit.trace(model, features)
    traced_model.save(args.traced_model)


