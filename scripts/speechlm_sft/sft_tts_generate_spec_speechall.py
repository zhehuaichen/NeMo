import json
import os
import random
import sys
import numpy as np
import torch
from tqdm import tqdm
from tts_normalization_utils import get_normalizer, normalize

from nemo.collections.asr.parts.preprocessing.features import clean_spectrogram_batch, normalize_batch
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel
from nemo.core.classes import typecheck

manifest_file = sys.argv[1]
manifest_name = os.path.basename(manifest_file).split(".")[0]
output_dir = f"./outputs_speechall/{manifest_name}/audios/"
normalize_type = "per_feature"
do_normalize = False
do_lowercase = False
use_enhancer = False  # TODO: fix it. It has a bug!
max_src_len = 256
sample_rate = 44100

# https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/tts_en_multispeaker_fastpitchhifigan
def generate_spec(
    tts_model, vocoder, text, normalizer=None, do_lowercase=False, enhancer_model=None, output_file_path=None
):
    original_len = len(text)
    if normalizer:
        text = normalize(text=text, normalizer=normalizer, do_lowercase=do_lowercase)

    src_ids = tts_model.parse(text, normalize=True)  # alternative tts_model.vocab.encode(text)

    with torch.no_grad():
        speaker_id = random.randint(0, n_speakers - 1)
        speaker_id = torch.tensor([speaker_id]).to(src_ids.device)
        spectrogram = tts_model.generate_spectrogram(tokens=src_ids, speaker=speaker_id)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        duration = audio.shape[1] / sample_rate
        # print(original_len, src_ids.shape, duration)

        # Save the audio to disk in a file called speech.wav
        sf.write(output_file_path, np.ravel(audio.to('cpu').numpy()), sample_rate, format='WAV')

    return audio, duration


tts_model = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker")
tts_model.eval().cuda()


# Load Vocoder
from nemo.collections.tts.models import HifiGanModel

vocoder = HifiGanModel.from_pretrained(model_name="tts_en_hifitts_hifigan_ft_fastpitch")
vocoder.eval().cuda()
n_speakers = tts_model.cfg.n_speakers

# Generate audio
import soundfile as sf

if use_enhancer:
    enhancer_model = SpectrogramEnhancerModel.from_pretrained(
        model_name="tts_en_spectrogram_enhancer_for_asr_finetuning"
    )
    enhancer_model.eval().cuda()
else:
    enhancer_model = None

if do_normalize:
    normalizer = get_normalizer()
else:
    normalizer = None

os.makedirs(output_dir, exist_ok=True)
max_num = 200000
tts_list = []
processed_tts_list = []
with open(manifest_file, 'r') as f:
    # filter according to len
    for i, line in enumerate(tqdm(f)):
        sample = json.loads(line)
        sample['context'] += ' . ' + sample["instruction"]
        if len(sample["context"]) < max_src_len:
            tts_list.append(sample)
    print(f"num of audios: {i}; after filtering {len(tts_list)}")
    for i, sample in enumerate(tqdm(tts_list)):
        sample = sample.copy()
        output_file_path = os.path.join(output_dir, sample["sample_id"]) + ".wav"
        sample['audio_filepath'] = os.path.join("./audios", sample["sample_id"]) + ".wav"
        sample['question'] = 'Transcribe and answer:'
        del sample['instruction']
        sample['text'] = sample['output']
        sample['answer'] = sample['output']
        del sample['output']

        try:
            audio, duration = generate_spec(
                tts_model,
                vocoder,
                text=sample["context"],
                normalizer=normalizer,
                do_lowercase=do_lowercase,
                enhancer_model=enhancer_model,
                output_file_path=output_file_path,
            )
            sample['duration'] = duration
            processed_tts_list.append(sample)
        except Exception as error:
            print(f"sample {sample} with error: {error}")

        if i >= max_num:
            break
content = ''
for sample in processed_tts_list:
    content += json.dumps(sample) + '\n'
final = open(output_dir + f'/../{manifest_name}.json', "w")
final.write(content)
final.close()
