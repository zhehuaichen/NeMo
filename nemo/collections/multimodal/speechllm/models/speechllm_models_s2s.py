

from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_model import MegatronT5SpeechLMModel
from typing import Dict, List, Optional, Union
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.utils import logging
import torch
import os
import json
import tempfile
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm.auto import tqdm
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging, model_utils
import torchaudio

# TODO(zhehuaic): change the inheritance to mcore model after adding LLM for
# joint finetuning
class ModularS2SModel(ASRModel):

    def _setup_generation_model(self, cfg: DictConfig, trainer: Trainer):
        self.generation_model = MegatronT5SpeechLMModel.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, trainer=trainer, cfg=cfg.model
            ).cuda()
        # a handle to reuse the data preprocessing of t5 tts
        self.generation_dataset = self.generation_model.get_dataset(dataset_paths=cfg.model.data.test_ds, for_train=False)

    def _setup_perception_model(self, cfg: DictConfig, map_location, trainer: Trainer):
        # restore ASR model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.stt_model.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        self.perception_model = imported_class.restore_from(
            restore_path=cfg.stt_model.model_path, map_location=map_location,
        )  # type: ASRModel
        self.perception_model.set_trainer(trainer)

    def __init__(self, cfg: DictConfig, trainer: Trainer, map_location=None):
        # TODO(zhehuaic): feed stt cfg for now because of inheriting ASRModel
        super().__init__(cfg=cfg.stt_model, trainer=trainer)
        self._setup_generation_model(cfg, trainer=trainer)
        self._setup_perception_model(cfg, map_location, trainer=trainer)
    
    
    def prepare_generation_input(self, batch, perception_output) -> Dict[str, torch.Tensor]:
        output_batch = []
        transform = torchaudio.transforms.Resample(16000, 22050).cuda()
        resampled_audios = transform(batch[0])
        resampled_length =  (batch[1] / 16000 * 22050).long()
        codecs = self.generation_model.encode_wav_from_codec_model(resampled_audios, resampled_length).cpu()
        for idx, hypotheses in enumerate(perception_output):
            input_example = {
                'question': f'Text to speech this {hypotheses}',
                'question_type': 'TEXT',
                # placeholder for answer fields
                'answer': codecs[idx],
                'answer_type': 'REFSPEAKERCODEC',
                'answer_duration': batch[1][idx]/16000,
                'taskname': 'squad',
                'lang': 'en',
                'context': codecs[idx],
                'context_type': 'REFSPEAKERCODEC',
                'context_duration': batch[1][idx]/16000,
            }            
            output_batch.append(self.generation_dataset.getitem_internal(input_example))
        output_batch = self.generation_dataset.collate_fn(output_batch)
        output_batch = [i.to(self.generation_model.device) for i in output_batch]
        return output_batch

    def validation_step(self, batch, batch_idx, dataloader_idx=0, default_batch_size=None):
        device=self.perception_model.device
        batch= [i.to(device) for i in batch]
        output_batch = self.perception_model.validation_step(batch, 0)
        beam_hypotheses = output_batch['translations']

        # extra processing for generation model
        generation_batch = self.prepare_generation_input(batch, beam_hypotheses)
        generation_result = self.generation_model.predict_step(generation_batch, batch_idx, dataset_batch_size=default_batch_size)
        # generation_result has been stored by t5 tts model
        # TODO(zhehuaic): move the audio store logic out
        return beam_hypotheses, generation_result

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: Union[List[str], str],
        batch_size: int = 4,
        logprobs: Optional[bool] = None,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        self.generation_model.eval()
        self.perception_model.eval()

        # get ready for new transcribe API
        if logprobs is not None:
            logging.warning("logprobs is deprecated, please use return_hypotheses instead")
            return_hypotheses = logprobs
        audio = paths2audio_files

        if audio is None or len(audio) == 0:
            return {}

        if return_hypotheses:
            logging.warning("return_hypotheses=True is currently not supported, returning text instead.")

        manifest_path = None
        if isinstance(audio, list):
            logging.debug(f"Found 'paths2audio_files' to be a list of {len(audio)} items.")
            logging.debug(f"Assuming each item in 'audio' is a path to audio file.")

            if isinstance(self.perception_model.tokenizer, tokenizers.AggregateTokenizer):
                primary_language = self.perception_model.tokenizer.langs[0]
                logging.debug(f"Transcribing with default setting of {primary_language}.")

        elif isinstance(audio, str):
            logging.debug(f"Found 'paths2audio_files' to be a string. Assuming it is a path to manifest file.")
            assert os.path.exists(audio), f"File {audio} doesn't exist"
            assert audio.endswith('.json') or audio.endswith('.jsonl'), f"File {audio} must be a json or jsonl file"

            # load json lines
            manifest_path = audio  # need to save this as we are overwriting paths2audio_files in nextline
            audio = manifest_utils.read_manifest(manifest_path)

        def _may_be_make_dict_and_fix_paths(json_items, manifest_path):
            out_json_items = []
            for item in json_items:
                if isinstance(item, str):
                    # assume it is a path to audio file
                    entry = {
                        'audio_filepath': item,
                        'duration': 100000,
                        'source_lang': 'en',
                        'taskname': 'asr',
                        'target_lang': 'en',
                        'pnc': 'yes',
                        'answer': 'nothing',
                    }
                elif isinstance(item, dict):
                    entry = item
                    entry['audio_filepath'] = get_full_path(entry['audio_filepath'], manifest_file=manifest_path)
                else:
                    raise ValueError(f"Expected str or dict, got {type(item)}")
                out_json_items.append(entry)
            return out_json_items

        paths2audio_files = _may_be_make_dict_and_fix_paths(audio, manifest_path)

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        try:
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        fp.write(json.dumps(audio_file) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self.perception_model._setup_transcribe_dataloader(config)
                for idx, test_batch in enumerate(tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose)):
                    beam_hypotheses, _ = self.validation_step(test_batch, idx, default_batch_size=batch_size)
                    hypotheses += beam_hypotheses
                    del test_batch
        finally:
            # set mode back to its original value
            logging.set_verbosity(logging_level)

        return hypotheses


    # TODO(zhehuaic): feed to stt for now because of inheriting ASRModel
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self.perception_model.setup_training_data(train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        self.perception_model.setup_validation_data(val_data_config)
