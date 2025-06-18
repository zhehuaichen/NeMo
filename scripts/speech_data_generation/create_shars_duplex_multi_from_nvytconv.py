import argparse
import copy
import csv
import glob
import json
import os
import random
import shutil
import wave
from io import BytesIO
from pathlib import Path

### from nemo.collections.tts.models import AudioCodecModel
import librosa
import numpy as np
import soundfile as sf
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import RecordingSet, save_audio
from lhotse.cut.base import Cut
from lhotse.features.base import Features, FeatureSet
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.shar.writers import AudioTarWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from nemo.utils import logging

#  python -m pdb -c continue /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex2/scripts/speech_data_generation/create_shars_duplex_multi_from_single.py --manifest /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.conversation_style_manifest_normalized_with_correctpath_with_evaluations.json.200 --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/tmp/msmarco_train_normalized.b.duplex.200/shars --num_shard 1


def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def divide_into_two_groups_with_similar_sum(numbers):
    """
    Divides a list of positive integers into two groups such that the
    sum of numbers in each group is as similar as possible.

    This function uses an exhaustive search approach, trying all possible
    ways to partition the list. It's suitable for lists of moderate size
    (e.g., up to 20-25 elements).

    Args:
        numbers (list[int]): A list of positive integers.

    Returns:
        tuple[list[int], list[int]]: A tuple containing two lists,
                                     representing the two groups.
                                     Returns ([], []) if the input list is empty.
    """
    n = len(numbers)
    if n == 0:
        return [], []

    min_diff = float('inf')
    best_group1 = []
    best_group2 = []

    # total_sum = sum(numbers) # Not strictly needed for this implementation path
    # but useful for understanding the target sum (total_sum / 2)

    # Iterate through all 2^n possible ways to form the first group.
    # Each integer 'i' from 0 to 2^n - 1 can be seen as a bitmask.
    # If the j-th bit of 'i' is set, numbers[j] goes into group1.
    # Otherwise, numbers[j] goes into group2.
    for i in range(1 << n):  # This loops from 0 to 2^n - 1
        current_group1 = []
        current_sum1 = 0
        current_group2 = []
        current_sum2 = 0

        for j in range(n):
            if (i >> j) & 1:  # Check if the j-th bit is set
                current_group1.append(j)
                current_sum1 += numbers[j]
            else:
                current_group2.append(j)
                current_sum2 += numbers[j]

        current_difference = abs(current_sum1 - current_sum2)

        if current_difference < min_diff:
            min_diff = current_difference
            best_group1 = current_group1
            best_group2 = current_group2

            # Optimization: If the difference is 0 (for even total sum)
            # or 1 (for odd total sum), we've found an optimal partition.
            # We can stop early, though the loop will find one such partition anyway.
            # total_sum_for_check = current_sum1 + current_sum2 # Should be sum(numbers)
            # if min_diff <= (total_sum_for_check % 2):
            #    return best_group1, best_group2

    return best_group1, best_group2


def create_shar_from_manifest(new_cuts, manifest, out_shar_dir, num_shard=10, segment_size=120, max_sub_seg_len=8):
    # manifest = "/lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/NV-YT-Conversations/manifest/09X2cpFfdJU.ndjson"
    in_manifest = {}
    for audio1_manifest in json_reader(manifest):
        audio1_name = audio1_manifest['audio_filepath']
        # assert audio1_name == audio1_manifest['speaker_reference']['speaker_audio_path']
        spkr = audio1_manifest['speaker_reference']['speaker_name']
        if spkr not in in_manifest:
            in_manifest[spkr] = [audio1_manifest]
        else:
            in_manifest[spkr].append(audio1_manifest)

    name_list = list(in_manifest.keys())
    num_list = [len(in_manifest[name]) for name in name_list]
    group1, group2 = divide_into_two_groups_with_similar_sum(num_list)
    group1_manifests = [in_manifest[name_list[i]] for i in group1]
    group2_manifests = [in_manifest[name_list[i]] for i in group2]

    def merge_manifests(manifests):
        merged_manifest = []
        for audio_manifest in manifests:
            merged_manifest.extend(audio_manifest)
        merged_manifest.sort(key=lambda point: point['start'])
        new_merged_manifest = []
        for segment in merged_manifest:
            start_time = segment.get('start')
            end_time = segment.get('end')
            if len(segment['text'].split()) / (end_time - start_time) > 20:
                print(
                    f"Problematic segment found: {segment['text']} {len(segment['text'].split())} with start {start_time} and end {end_time}. Too many words per second."
                )
                continue

            if (
                len(new_merged_manifest) > 0
                and new_merged_manifest[-1]['speaker'] == segment['speaker']
                and new_merged_manifest[-1]['end'] - new_merged_manifest[-1]['start'] < max_sub_seg_len
            ):
                assert new_merged_manifest[-1]['end'] < segment['start']
                new_merged_manifest[-1]['end'] = segment['end']
                new_merged_manifest[-1]['text'] += ' ' + segment['text']
            else:
                new_merged_manifest.append(segment)
        return new_merged_manifest

    group1_manifests = merge_manifests(group1_manifests)
    group2_manifests = merge_manifests(group2_manifests)

    def get_min_start_time(group_manifests):
        min_start_time = float('inf')
        for segment in group_manifests:
            if segment['start'] < min_start_time:
                min_start_time = segment['start']
        return min_start_time

    if get_min_start_time(group1_manifests) < get_min_start_time(group2_manifests):
        user_manifest = group1_manifests
        agent_manifest = group2_manifests
    else:
        user_manifest = group2_manifests
        agent_manifest = group1_manifests

    def get_wav(audio_name):
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(audio_name)
        if data.ndim > 1:
            # If stereo, convert to mono by averaging channels
            data = data[:, 0]
        return data, sample_rate

    def _process_manifest_segments(
        manifest, original_audio_data, output_audio_array, sample_rate, total_samples, role_name="role"  # For logging
    ):
        """
        Helper function to process audio segments based on a manifest.

        Args:
            manifest (list): List of segment dictionaries {'start': float, 'end': float}.
            original_audio_data (np.ndarray): The source audio data.
            output_audio_array (np.ndarray): The numpy array to copy segments into.
            sample_rate (int): Sample rate of the audio.
            total_samples (int): Total number of samples in the original audio.
            role_name (str): Name of the role (e.g., "user", "agent") for logging.
        """
        # print(f"\nProcessing {role_name} manifest...")
        for i, segment in enumerate(manifest):
            start_time = segment.get('start')
            end_time = segment.get('end')

            # Validate segment data
            if start_time is None or end_time is None:
                print(f"Warning: Segment {i} in {role_name}_manifest is missing 'start' or 'end'. Skipping.")
                continue
            if not (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))):
                print(f"Warning: Segment {i} in {role_name}_manifest has non-numeric 'start' or 'end'. Skipping.")
                continue
            if start_time >= end_time:
                print(
                    f"Warning: Segment {i} in {role_name}_manifest has start_time ({start_time}) >= end_time ({end_time}). Skipping."
                )
                continue

            # Convert start and end times from seconds to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Ensure indices are within the bounds of the audio data
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)

            # Check again after clamping to ensure segment is still valid
            if start_sample >= end_sample:
                print(
                    f"Warning: Segment {i} in {role_name}_manifest results in zero or negative duration after clamping. Skipping."
                )
                continue

            # print(f"  {role_name.capitalize()} segment {i+1}: {start_time:.2f}s - {end_time:.2f}s -> samples {start_sample} - {end_sample}")

            # Copy the audio segment from original to the specified output_audio_array
            output_audio_array[start_sample:end_sample] = original_audio_data[start_sample:end_sample]

    def split_audio_by_manifest(
        original_audio_data,
        sample_rate,
        user_manifest,
        agent_manifest,
        user_output_filename="user_audio.wav",
        agent_output_filename="agent_audio.wav",
    ):
        """
        Splits an audio file into two separate files for user and agent based on manifests.

        Args:
            original_audio_data (np.ndarray): NumPy array containing the original audio data.
                                            Assumed to be mono.
            sample_rate (int): The sample rate of the audio in Hz.
            user_manifest (list): A list of dictionaries, where each dictionary has
                                'start' and 'end' keys with time in seconds.
                                Example: [{'start': 0.5, 'end': 1.5}, ...]
            agent_manifest (list): Similar to user_manifest, but for the agent.
            user_output_filename (str): Filename for the output user audio.
            agent_output_filename (str): Filename for the output agent audio.
        """
        if original_audio_data is None or sample_rate is None:
            print("Error: Original audio data or sample rate is not provided.")
            return

        # Ensure audio is 1D (mono)
        if original_audio_data.ndim > 1:
            print("Warning: Original audio has multiple channels. Converting to mono by averaging.")
            original_audio_data = np.mean(original_audio_data, axis=1)

        # Get the total number of samples in the original audio
        total_samples = len(original_audio_data)

        # Create silent audio arrays for user and agent
        # These will have the same duration and sample rate as the original
        user_audio_output = np.zeros(total_samples, dtype=original_audio_data.dtype)
        agent_audio_output = np.zeros(total_samples, dtype=original_audio_data.dtype)

        # Process user manifest using the helper function
        _process_manifest_segments(
            manifest=user_manifest,
            original_audio_data=original_audio_data,
            output_audio_array=user_audio_output,
            sample_rate=sample_rate,
            total_samples=total_samples,
            role_name="user",
        )

        # Process agent manifest using the helper function
        _process_manifest_segments(
            manifest=agent_manifest,
            original_audio_data=original_audio_data,
            output_audio_array=agent_audio_output,
            sample_rate=sample_rate,
            total_samples=total_samples,
            role_name="agent",
        )

        return user_audio_output, agent_audio_output, sample_rate

    audio1, sample_rate1 = get_wav(audio1_name)
    # given user_manifest and agent_manifest to create 2 audio files corresponding to each. When one role is talking, the other role should be silent. The information about each role is specified in the 'start' and 'end' of each segment.
    user_audio, agent_audio, sample_rate = split_audio_by_manifest(
        original_audio_data=audio1,
        sample_rate=sample_rate1,
        user_manifest=user_manifest,
        agent_manifest=agent_manifest,
        user_output_filename="output_user_speech.wav",
        agent_output_filename="output_agent_speech.wav",
    )
    segment_i_start = 0
    num_skip = 0
    num_keep = 0
    while segment_i_start < user_manifest[-1]['start']:

        def get_first_segment(segments, start_time):
            for segment in segments:
                if segment['start'] >= start_time:
                    return segment
            return None

        segment_i_start = get_first_segment(user_manifest, segment_i_start)['start']
        assert segment_i_start is not None
        segment_i_end = segment_i_start + segment_size

        def get_segements_between(segments, start_time, end_time, role):
            result = []
            for segment in segments:
                if segment['start'] >= start_time and segment['end'] <= end_time:
                    segment['duration'] = segment['end'] - segment['start']
                    segment['channel'] = 0
                    segment['language'] = 'EN'
                    segment['real_speaker'] = segment['speaker']
                    segment['speaker'] = role
                    result.append(segment)
            return result

        user_segments = get_segements_between(user_manifest, segment_i_start, segment_i_end, 'user')
        agent_segments = get_segements_between(agent_manifest, segment_i_start, segment_i_end, 'agent')
        if len(user_segments) <= 0 or len(agent_segments) <= 0:
            # print(f"skip {segment_i_start} {segment_i_end} {len(user_segments)} {len(agent_segments)}")
            num_skip += 1
            segment_i_start += segment_size // 2
            continue

        def get_step_size(dur):
            return int(dur * sample_rate1)

        segment_i_end = max(user_segments[-1]['end'], agent_segments[-1]['end'])
        assert segment_i_end - segment_i_start <= segment_size

        if len(user_audio) < get_step_size(segment_i_end):
            user_audio = np.pad(user_audio, (0, get_step_size(segment_i_end) - len(user_audio)))
            agent_audio = np.pad(agent_audio, (0, get_step_size(segment_i_end) - len(agent_audio)))

        # TODO: produce an example for every 60 sec chunk
        new_cut = MonoCut(
            id=f"{os.path.basename(audio1_name)}_{segment_i_start}",
            start=0,
            duration=segment_i_end - segment_i_start,
            channel=0,
            supervisions=[],
        )

        def offset_segments(segments, offset):
            segments = copy.deepcopy(segments)
            for segment in segments:
                segment['start'] -= offset
                segment['end'] -= offset
            return segments

        user_stream = BytesIO()
        agent_stream = BytesIO()

        save_audio(
            dest=user_stream,
            src=user_audio[get_step_size(segment_i_start) : (get_step_size(segment_i_end) + 1)],
            sampling_rate=sample_rate1,
            format="wav",
        )
        save_audio(
            dest=agent_stream,
            src=agent_audio[get_step_size(segment_i_start) : (get_step_size(segment_i_end) + 1)],
            sampling_rate=sample_rate1,
            format="wav",
        )
        user_stream.seek(0)
        agent_stream.seek(0)
        new_cut.recording = Recording.from_bytes(user_stream.getvalue(), f"{new_cut.id}_user")
        new_cut.target_audio = Recording.from_bytes(agent_stream.getvalue(), f"{new_cut.id}_agent")
        new_cut.user_segments = offset_segments(user_segments, segment_i_start)
        new_cut.agent_segments = offset_segments(agent_segments, segment_i_start)
        segment_i_start += segment_size // 2
        new_cuts.append(new_cut)
        num_keep += 1
    print(f"skip {num_skip} keep {num_keep} num_shards {num_shard}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=str,
        default=None,  # "/lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/NV-YT-Conversations/manifest/09X2cpFfdJU.ndjson"
    )
    parser.add_argument(
        '--manifest_path',
        type=str,
        default=None,  # "/lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/NV-YT-Conversations/manifest/"
    )
    parser.add_argument(
        '--manifest_list',
        type=str,
        default=None,  # ""
    )
    parser.add_argument(
        '--out_shar_dir',
        type=str,
        default="/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/s2s_lhotse_with_wavs/squadv2/",
    )
    parser.add_argument(
        '--num_shard',
        type=int,
        default=10,
    )

    args = parser.parse_args()
    print(f"out_shar_dir {args.out_shar_dir}")
    print(f"num_shard {args.num_shard}")

    if args.manifest is not None:
        manifest_list = [args.manifest]
    elif args.manifest_path is not None:
        manifest_list = sorted(glob.glob(os.path.join(args.manifest_path, "*.ndjson")))
        if len(manifest_list) == 0:
            raise ValueError(f"No manifest files found in {args.manifest_path}")
    elif args.manifest_list is not None:
        manifest_list = open(args.manifest_list).readlines()
        manifest_list = [line.strip() for line in manifest_list if line.strip()]
        if len(manifest_list) == 0:
            raise ValueError(f"No manifest files found in {args.manifest_list}")
    new_cuts = []
    c = 0
    for manifest in manifest_list:
        try:
            print(f"Processing manifest {c} {manifest}")
            create_shar_from_manifest(
                new_cuts=new_cuts,
                manifest=manifest,
                out_shar_dir=args.out_shar_dir,
                num_shard=args.num_shard,
            )
            c += 1
            # if c == 10:
            #    break
        except Exception as e:
            print(f"Error processing manifest {manifest}: {e}")
            continue
    cuts = CutSet(cuts=new_cuts)
    shard_size = int(len(new_cuts) / args.num_shard)
    if len(new_cuts) % shard_size != 0:
        shard_size += 1

    print(f"...Making Shars {len(cuts)} cuts, shard_size {shard_size}")
    out_shar_dir = Path(args.out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(
        out_shar_dir, fields={"recording": "flac", "target_audio": "flac"}, num_jobs=1, shard_size=shard_size
    )
    print(f"...share created")


if __name__ == "__main__":
    main()
