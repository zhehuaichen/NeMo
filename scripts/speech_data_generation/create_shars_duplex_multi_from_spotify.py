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


import logging

import numpy as np
import sphn
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def read_audio(audio_meta: dict, target_sampling_rate: int) -> np.ndarray | None:
    """
    Reads and resamples a segment of an audio file into a mono NumPy array.

    This function reads a specified segment from an audio file, converts it to a
    single (mono) channel, and resamples it to the target rate. It returns the
    audio as a normalized floating-point NumPy array.

    The function first tries reading with `sphn`. If it fails, it falls back to
    `pydub`, which supports a wider range of audio formats.

    Args:
        audio_meta: A dictionary containing audio metadata with the keys:
                    'path' (str): The path to the audio file.
                    'offset' (float): The start offset in seconds.
                    'duration' (float): The duration to read in seconds.
        target_sampling_rate: The desired sampling rate for the output audio.

    Returns:
        A mono NumPy array of the audio waveform shaped as `(1, n_samples)`
        with values normalized between -1.0 and 1.0. Returns `None` if
        reading fails with both methods.
    """
    try:
        array = np.mean(
            sphn.read(
                audio_meta["path"],
                start_sec=audio_meta["offset"],
                duration_sec=audio_meta["duration"],
                sample_rate=target_sampling_rate,
            )[0],
            0,
            keepdims=True,
        )
    except Exception as e:
        logger.info(
            f"Failed to read audio with `sphn`:\n"
            f"\tpath: {audio_meta['path']}\n"
            f"\tstart_sec: {audio_meta['offset']}, duration_sec: {audio_meta['duration']}\n"
            f"\tError: {e}\n"
            f"Attempting with `pydub`."
        )
        try:
            audio_segment = AudioSegment.from_file(
                audio_meta["path"], start_second=audio_meta["offset"], duration=audio_meta["duration"]
            ).set_frame_rate(target_sampling_rate)
            array = np.array(audio_segment.get_array_of_samples())
            if audio_segment.channels > 1:
                array = array.reshape((-1, audio_segment.channels)).T
            else:
                array = array[np.newaxis, :]
            array = np.mean(array, 0, keepdims=True)
        except Exception as e2:
            logger.info(
                f"Failed to read audio with `pydub`:\n"
                f"\tpath: {audio_meta['path']}\n"
                f"\tstart_second: {audio_meta['offset']}, duration: {audio_meta['duration']}\n"
                f"\tError: {e2}\n"
                f"Skipping."
            )
            return
    return array


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


def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str = "", aws_profile: str = None):
    """
    Uploads all files from a local directory to a specified S3 bucket.

    Args:
        local_directory (Path): The path to the local directory containing files to upload.
        s3_bucket (str): The name of the S3 bucket.
        s3_prefix (str): An optional S3 prefix (folder path) where files will be uploaded.
                         e.g., 'my_shar_archives/data/'.
        aws_profile (str): Optional. The AWS profile name from your ~/.aws/credentials file.
                           If None, boto3 will use default credentials (e.g., environment variables).
    """
    if not local_directory.is_dir():
        print(f"Error: Local directory '{local_directory}' does not exist.")
        return

    try:
        # Initialize S3 client, optionally with a specific profile
        import boto3

        s3_client = boto3.client('s3')

        # Walk through local directory and upload files
        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_path = Path(root) / filename
                # Construct the S3 key: s3_prefix + relative_path_from_local_directory
                relative_path = local_path.relative_to(local_directory)
                s3_key = str(Path(s3_prefix) / relative_path)

                print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
                s3_client.upload_file(str(local_path), s3_bucket, s3_key)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def create_shar_from_manifest(
    start_idx,
    end_idx,
    tmp_dir,
    new_cuts,
    manifest,
    out_shar_dir,
    num_shard=10,
    segment_size=120,
    max_sub_seg_len=8,
    max_sub_gap_len=1,
):
    # manifest = "/lustre/fsw/portfolios/adlr/projects/adlr_audio_speech/datasets/NV-YT-Conversations/manifest/09X2cpFfdJU.ndjson"
    for processed, audio1_manifest in enumerate(json_reader(manifest)):
        if processed < start_idx or processed >= end_idx:
            continue
        in_manifest = {}
        audio1_name = audio1_manifest['audio_path']
        for segment in audio1_manifest['segments']:
            # assert audio1_name == audio1_manifest['speaker_reference']['speaker_audio_path']
            if 'speaker' not in segment:
                continue
            spkr = segment['speaker']
            if spkr not in in_manifest:
                in_manifest[spkr] = [segment]
            else:
                in_manifest[spkr].append(segment)

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
                    # print(f"Problematic segment found: {segment['text']} {len(segment['text'].split())} with start {start_time} and end {end_time}. Too many words per second.")
                    continue

                if (
                    len(new_merged_manifest) > 0
                    and new_merged_manifest[-1]['speaker'] == segment['speaker']
                    and new_merged_manifest[-1]['end'] - new_merged_manifest[-1]['start'] < max_sub_seg_len
                    and segment['start'] - new_merged_manifest[-1]['end'] < max_sub_gap_len
                    and new_merged_manifest[-1]['end'] < segment['start']
                ):
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
            import sphn

            data, sample_rate = sphn.read(audio_name)
            if data.ndim > 1:
                # If stereo, convert to mono by averaging channels
                data = data[0]
            return data, sample_rate

        def _process_manifest_segments(
            manifest,
            original_audio_data,
            output_audio_array,
            sample_rate,
            total_samples,
            role_name="role",  # For logging
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

            user_stream = tmp_dir + f"/{os.path.basename(audio1_name)}_{segment_i_start}" + "_user_audio.wav"
            agent_stream = tmp_dir + f"/{os.path.basename(audio1_name)}_{segment_i_start}" + "_agent_audio.wav"

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
            new_cut.recording = Recording.from_file(user_stream, f"{new_cut.id}_user")
            new_cut.target_audio = Recording.from_file(agent_stream, f"{new_cut.id}_agent")
            new_cut.user_segments = offset_segments(user_segments, segment_i_start)
            new_cut.agent_segments = offset_segments(agent_segments, segment_i_start)
            segment_i_start += segment_size // 2
            new_cuts.append(new_cut)
            num_keep += 1
        print(f"processed {processed} skip {num_skip} keep {num_keep} num_shards {num_shard}")
        # if processed % 10 == 9:
        #     break


import ctypes  # For interrupting threads (Windows specific, and not truly reliable)
import threading
import time


class FunctionTimeout(Exception):
    """Custom exception raised when a function execution times out."""

    pass


def _async_raise(tid, exctype):
    """Raises an exception in the specified thread."""
    if not isinstance(exctype, type):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res != 1:
        # "if res > 1" in newer versions of python
        # Pre-emptive clean up
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def raise_exception_in_thread(target_thread_id, exception):
    """Raises an exception in a given thread using its ID."""
    _async_raise(target_thread_id, exception)


import multiprocessing
import os
import time


def execute_with_timeout(func, args=(), kwargs={}, timeout=120):
    """
    Executes a function in a separate process with a timeout.

    Args:
        func (callable): The function to execute.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.
        timeout (int): The maximum execution time in seconds.

    Returns:
        Any: The return value of the function if it completes within the timeout,
             or None if it times out.
    """
    # Use a multiprocessing.Queue to get the result back from the child process
    result_queue = multiprocessing.Queue()

    def target_function(q, *a, **kw):
        try:
            res = func(*a, **kw)
            try:
                q.put(res)
            except BrokenPipeError:
                print(f"[{os.getpid()}] Broken pipe detected when putting result. Parent probably terminated.")
        except Exception as e:
            try:
                q.put(e)  # Put the exception in the queue if one occurs
            except BrokenPipeError:
                print(f"[{os.getpid()}] Broken pipe detected when putting exception. Parent probably terminated.")

    # Create a Process object
    process = multiprocessing.Process(target=target_function, args=(result_queue,) + args, kwargs=kwargs)

    # print(f"Starting function '{func.__name__}' in a new process (PID: {process.pid})...")
    process.start()

    process.join(timeout=timeout)  # Wait for the process to complete, with a timeout

    if process.is_alive():
        print(f"Function '{func.__name__}' timed out after {timeout} seconds. Terminating process {process.pid}.")
        process.terminate()  # Terminate the process
        process.join()  # Wait for the process to actually terminate
        print(f"Process {process.pid} terminated.")
        return None  # Indicate that it timed out
    else:
        # If the process is not alive, it either completed or crashed
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"Function '{func.__name__}' raised an exception: {result}")
                raise result  # Re-raise the exception for the caller
            else:
                print(f"Function '{func.__name__}' completed successfully within {timeout} seconds.")
                return result
        else:
            # This case means the process finished but didn't put anything in the queue,
            # perhaps due to an unexpected exit or error in the child process before putting result.
            print(f"Function '{func.__name__}' process finished but no result was retrieved.")
            return None  # Or raise a specific error indicating an unexpected exit


def is_file_valid(filepath):
    """
    Checks if a file exists and is not empty.
    """
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


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
    parser.add_argument(
        '--time_out',
        type=int,
        default=16,
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
    with multiprocessing.Manager() as manager:
        c = 0
        for manifest in manifest_list:
            num_lines = sum(1 for _ in open(manifest))
            batch_size = 200
            import tempfile

            for start_idx in range(0, num_lines, batch_size):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    end_idx = min(start_idx + batch_size, num_lines)
                    print(f"Processing manifest {c} {manifest} from line {start_idx} to {end_idx}")
                    out_shar_dir = Path(args.out_shar_dir + f'/{start_idx}')
                    if is_file_valid(str(out_shar_dir) + "/cuts.000000.jsonl.gz"):
                        print(
                            f"Output shar directory {out_shar_dir} already exists and contains cuts.000000.jsonl.gz. Exiting to avoid overwriting."
                        )
                        continue

                    new_cuts = manager.list()
                    new_cuts = []
                    try:
                        create_shar_from_manifest(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            tmp_dir=tmp_dir,
                            new_cuts=new_cuts,
                            manifest=manifest,
                            out_shar_dir=args.out_shar_dir,
                            num_shard=args.num_shard,
                        )
                        # result_thread1 = execute_with_timeout(create_shar_from_manifest, kwargs={
                        #     'start_idx': start_idx, 'end_idx': end_idx,
                        #     'tmp_dir': None, 'new_cuts':new_cuts, 'manifest':manifest, 'out_shar_dir':args.out_shar_dir, 'num_shard':args.num_shard,
                        #     }, timeout=args.time_out)
                        c += 1
                        # if c == 10:
                        #     break
                    except Exception as e:
                        print(f"Error processing manifest {manifest}: {e}")
                        continue

                    shard_size = int(len(new_cuts) / args.num_shard)
                    print(f"...Making Shars {len(new_cuts)} cuts, shard_size {shard_size}")
                    cuts = CutSet(cuts=new_cuts)
                    if len(new_cuts) % shard_size != 0:
                        shard_size += 1

                    out_shar_dir.mkdir(parents=True, exist_ok=True)
                    # assert len(user_recordings) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
                    exported = cuts.to_shar(
                        out_shar_dir,
                        fields={"recording": "flac", "target_audio": "flac"},
                        num_jobs=1,
                        shard_size=shard_size,
                    )
            if 0:
                upload_directory_to_s3(
                    local_directory=out_shar_dir,
                    s3_bucket='data',
                    s3_prefix=str(out_shar_dir).split('data/')[-1],
                )

        print(f"...share created: {out_shar_dir}")


if __name__ == "__main__":
    main()
