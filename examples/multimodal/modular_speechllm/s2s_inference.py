# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.core.config import hydra_runner
import logging
from nemo.collections.multimodal.speechllm.models.speechllm_models_s2s import ModularS2SModel
    

"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  compute_timestamps: Bool to request greedy time stamp information (if the model supports it)
  compute_langs: Bool to request language ID information (if the model supports it)

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  (Optionally: You can limit the type of timestamp computations using below overrides)
  ctc_decoding.ctc_timestamp_type="all"  # (default all, can be [all, char, word])
  rnnt_decoding.rnnt_timestamp_type="all"  # (default all, can be [all, char, word])

  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_transcripts: Bool which when set allows repeated transcriptions to overwrite previous results.

  ctc_decoding: Decoding sub-config for CTC. Refer to documentation for specific values.
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

  calculate_wer: Bool to decide whether to calculate wer/cer at end of this script
  clean_groundtruth_text: Bool to clean groundtruth text
  langid: Str used for convert_num_to_words during groundtruth cleaning
  use_cer: Bool to use Character Error Rate (CER)  or Word Error Rate (WER)

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
append_pred - optional. Allows you to add more than one prediction to an existing .json
pred_name_postfix - optional. The name you want to be written for the current model
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    clean_groundtruth_text=True \
    langid='en' \
    batch_size=32 \
    compute_timestamps=False \
    compute_langs=False \
    cuda=0 \
    amp=True \
    append_pred=False \
    pred_name_postfix="<remove or use another model name for output filename>"
"""

@hydra_runner(config_path="conf", config_name="s2s_t5_inference.yaml")
def main(s2s_cfg):
    cfg = s2s_cfg.stt_model
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    logging.info(f'Hydra config: {OmegaConf.to_yaml(s2s_cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    # MegatronTrainerBuilder compat checks
    if "gradient_as_bucket_view" not in s2s_cfg.model:
        with open_dict(s2s_cfg):
            s2s_cfg.model.gradient_as_bucket_view=False
    trainer = MegatronTrainerBuilder(s2s_cfg).create_trainer()
    exp_manager(trainer, s2s_cfg.exp_manager)
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(s2s_cfg):
        s2s_cfg.model.precision = s2s_cfg.trainer.precision
    model = ModularS2SModel(s2s_cfg, trainer=trainer)

    model = model.eval()
    asr_model = model.perception_model
    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if isinstance(asr_model.decoding, MultiTaskDecoding):
            asr_model.change_decoding_strategy(cfg.multitask_decoding)
        else:
            raise ValueError(f"Decoding strategy is not supported for {type(asr_model.decoding)}")

    # prepare audio filepaths and decide wether it's partial audio
    filepaths, partial_audio = prepare_audio_data(cfg)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast(dtype=None):
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model.perception_model.__class__.__name__)

    # transcribe audio
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    with autocast(dtype=amp_dtype):
        with torch.no_grad():
            transcriptions = model.transcribe(
                paths2audio_files=filepaths,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                return_hypotheses=cfg.return_hypotheses,
                channel_selector=cfg.channel_selector,
                augmentor=augmentor,
            )

    logging.info(f"Finished transcribing {len(filepaths)} files !")
    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
    if type(transcriptions) == tuple and len(transcriptions) == 2:
        transcriptions = transcriptions[0]

    # write audio transcriptions
    output_filename, pred_text_attr_name = write_transcription(
        transcriptions,
        cfg,
        f"{model.perception_model.__class__.__name__}",
        filepaths=filepaths,
        compute_langs=False,
        compute_timestamps=False,
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

    return transcriptions


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
