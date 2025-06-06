import math
import random
import re

import numpy as np
import torch.utils.data
import torchaudio
from lhotse import CutSet, Recording
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    TextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)
from nemo.utils import logging


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        context_key: str = "context",
        default_context_key: str = "default_context",
        vocab_sizes: list[int] = [-1],
        decoder_reduction_factor: int = 1,
        speech_pad_id: int = 1001,
        speech_unk_id: int = 1002,
        speech_bos_id: int = 1003,
        speech_eos_id: int = 1004,
        filter_by_source_target_text_ratio: bool = False,
        source_target_text_ratio_limit: float = 1.0,
        codec_sample_rate: int = 22050,
        sample_rate: int = 16000,
        t5_style: bool = False,
        load_answer_audio: bool = False,
        codec_model_downsampling_factor: float = 1023.5,
        prompt_audio_path: str = None,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key

        if len(vocab_sizes) == 1 and vocab_sizes[0] <= 0:
            vocab_sizes = [self.text_processor.tokenizer.vocab_size]
        self.vocab_sizes = list(vocab_sizes)
        self.n_speech_codebooks = len(self.vocab_sizes) - 1
        self.decoder_reduction_factor = decoder_reduction_factor
        self.speech_pad_id = speech_pad_id
        self.speech_unk_id = speech_unk_id
        self.speech_bos_id = speech_bos_id
        self.speech_eos_id = speech_eos_id
        self.filter_by_source_target_text_ratio = filter_by_source_target_text_ratio
        self.source_target_text_ratio_limit = source_target_text_ratio_limit
        self.codec_sample_rate = codec_sample_rate
        self.sample_rate = sample_rate
        self.load_answer_audio = load_answer_audio
        self.codec_model_downsampling_factor = codec_model_downsampling_factor

        # To be consistent with SALM text processor
        self.text_processor.add_sep = False
        self.text_processor.max_seq_length = (
            4096  # Set this to a large number for since the speech sequence can be long
        )
        self.t5_style = t5_style
        if self.codec_sample_rate != self.sample_rate:
            logging.info(f'{self.codec_sample_rate} {self.sample_rate} are different')

        if prompt_audio_path is not None:
            audio, sr = torchaudio.load(prompt_audio_path)
            self.prompt_audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

    def _extract_text_and_time_tokens(self, input_sequence):
        # Regular expression to match time tokens (e.g., <|x|> where x is an integer)
        time_token_pattern = r"<\|(\d+)\|>"
        # Find all time tokens
        time_tokens = re.findall(time_token_pattern, input_sequence)
        # Only keep the first token of every pair (i.e., start time tokens)
        start_time_token = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
        # Remove all time tokens to isolate words
        words = re.sub(time_token_pattern, '', input_sequence).split()
        # Process each word, tokenize it, and calculate token lengths
        tokenized_words = []
        word_length = []
        for idx, word in enumerate(words):
            # Tokenize the word using the provided text processor
            tokenized_word = self.text_processor._process_example(context="", output=word)
            # Remove the EOS token (assuming the EOS token is at the end of "answer_ids")
            token_ids = tokenized_word["answer_ids"][:-1]  # Remove EOS token
            if idx != 0:  # If not the first word, remove the first token
                token_ids = token_ids[1:]
            token_length = len(token_ids)  # Calculate the length
            tokenized_words.extend(token_ids)
            word_length.append(token_length)
        return (
            torch.as_tensor(tokenized_words),
            torch.as_tensor(start_time_token),
            torch.as_tensor(word_length),
        )

    def _expand_text_with_timestamps_and_word_lengths(
        self, word_tokens, word_lengths, start_time_tokens, features_lens, frame_rate=0.08, pad_id=None
    ):
        """
        Expand word tokens according to start time tokens and word lengths for a batch of sequences.

        Args:
        - word_tokens: List of lists of token sequences (each inner list is a word's token IDs), shape [batch][time].
        - word_lengths: List of lists of word lengths, shape [batch][time].
        - start_time_tokens: List of lists of start times, shape [batch][time].
        - max_length: Maximum length in the time dimension (number of frames).
        - frame_rate: Frame rate resolution.
        - pad_id: Padding ID to use for empty positions in the tensor.

        Returns:
        - 2D tensor [batch, max_length] where each row is the expanded token sequence for that batch.
        """

        def discretize_time(start_token, speech_resolution=0.08, timestamp_resolution=0.08):
            """Convert the start token into a time index based on the resolution."""
            return int(start_token * timestamp_resolution / speech_resolution)

        if pad_id is None:
            raise ValueError("pad_id must be provided.")

        batch_size = len(word_tokens)
        max_length = max(features_lens).item()

        # Create the empty 2D tensor [batch, max_length] with pad_id as the default value
        texts_expanded = torch.full((batch_size, max_length), fill_value=pad_id, dtype=torch.long)

        # Iterate over each batch
        for batch_idx in range(batch_size):
            # Remove the speech eos
            batch_max_length = features_lens[batch_idx] - 1
            word_start_idx = 0  # Start index to keep track of the position within the concatenated word tokens

            for word_idx, word_length in enumerate(word_lengths[batch_idx]):
                start_token = start_time_tokens[batch_idx][word_idx]

                # Convert the start time token into a time index based on frame rate
                start_time_index = discretize_time(start_token, frame_rate)

                # Reduction of start time index due to stacking of frames
                start_time_index = int(start_time_index / self.decoder_reduction_factor)

                end_time_index = start_time_index + word_length
                end_time_index = min(end_time_index, max_length)

                # Get the word tokens for the current word
                word_token_ids = word_tokens[batch_idx][word_start_idx : word_start_idx + word_length]

                # Populate the tokens in the expanded tensor at the correct positions
                for t_idx in range(start_time_index, end_time_index):
                    if t_idx - start_time_index < len(word_token_ids):  # Ensure tokens are within bounds
                        token_id = word_token_ids[t_idx - start_time_index]  # Get token for this time step
                        texts_expanded[batch_idx][t_idx] = token_id  # Directly assign the token ID

                # Move to the next word in the concatenated word tokens
                word_start_idx += word_length

            # Overwrite padding tokens
            texts_expanded[batch_idx][batch_max_length:] = self.text_processor.pad_id
        return texts_expanded

    def __getitem__duplex_(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        import re

        cuts = cuts.sort_by_duration()

        metadata = []
        instructions, instruction_lengths = [], []
        target_texts, target_text_lengths = [], []
        source_texts, source_text_lengths = [], []
        start_time_tokens, word_lengths = [], []
        num_turns = []
        text_start_time = []
        text_end_time = []
        skipped_source = 0
        processed_cuts = []
        for cut in cuts:
            cut.system_prompt = [
                sup for sup in cut.supervisions if sup.duration == 0.0 and sup.start == 0.0
            ]  # ignore system prompt
            assert len(cut.system_prompt) <= 1, f"More than one system prompt in {cut}"
            cut.system_prompt = cut.system_prompt[0].text if len(cut.system_prompt) > 0 else ''
            cut.supervisions = [sup for sup in cut.supervisions if sup.duration > 0.0]  # ignore system prompt

        def get_duration_from_sample(field):
            return field.num_samples / field.sampling_rate

        for cut in cuts:
            if np.isclose(cut.target_audio.duration, cut.recording.duration) and np.isclose(
                get_duration_from_sample(cut.target_audio), get_duration_from_sample(cut.recording)
            ):
                is_valid = True
            else:
                is_valid = False
            # filter problematic cuts with close to empty transcriptions
            if len((" ".join([sup.text for sup in cut.supervisions])).split()) < len(cut.supervisions):
                is_valid = False
            for id, sup in enumerate(cut.supervisions):
                if id % 2 == 0:
                    if sup.speaker.lower() != "user":
                        is_valid = False
                else:
                    if sup.speaker.lower() != "agent" and sup.speaker.lower() != "assistant":
                        is_valid = False
            if is_valid:
                processed_cuts.append(cut)
            else:
                logging.info(f"Skipping cut {cut}")
        cuts = CutSet(cuts=processed_cuts)
        for id, cut in enumerate(cuts):

            def validate_time(input_time):
                if input_time > cut.duration + 0.16:
                    logging.info(f"{input_time} > {cut.duration} in {cut}")
                return min(input_time, cut.duration)

            num_turns.append(len(cut.supervisions))
            metadata.append({'audio_filepath': cut.id + '.wav'})
            text_start_time.append([])
            text_end_time.append([])
            # treat multiturn data as multiple batch each with 2-turn conversation
            for i in range(0, len(cut.supervisions), 2):
                supervisions = cut.supervisions[i : i + 2]
                # TODO: the following use of _process_example is not ideal. Should update
                if supervisions[0].speaker.lower() != "user":
                    logging.info(f"First speaker should be user {cut}")
                instruction = self.text_processor._process_example(context=supervisions[0].text, output="")
                instruction, instruction_length = torch.as_tensor(instruction["input_ids"][:-1]), torch.as_tensor(
                    len(instruction["input_ids"]) - 1
                )
                # Extract user text
                pattern = r"<\|\d+\|>"
                output_text = re.sub(pattern, "", supervisions[0].text)
                output_text = re.sub(r'\s+', ' ', output_text).strip()
                source_text = self.text_processor._process_example(context="", output=output_text)
                # -1 to remove the eos token added by the text processor
                source_text, source_text_length = torch.as_tensor(source_text["answer_ids"][:-1]), torch.as_tensor(
                    len(source_text["answer_ids"]) - 1
                )

                if len(supervisions) <= 1:
                    instructions.append(instruction)
                    instruction_lengths.append(instruction_length)
                    source_texts.append(source_text)
                    source_text_lengths.append(source_text_length)
                    text_start_time[-1].append(supervisions[0].start + supervisions[0].duration)
                    text_end_time[-1].append(supervisions[0].start + supervisions[0].duration)
                    text_start_time[-1][-1] = validate_time(text_start_time[-1][-1])
                    text_end_time[-1][-1] = validate_time(text_end_time[-1][-1])
                    skipped_source += 1
                    continue
                elif supervisions[1].speaker.lower() == "agent" or supervisions[1].speaker.lower() == "assistant":
                    use_word_alignment = getattr(cut, "s2s_duplex_align", False)
                    text = supervisions[1].text
                    if not use_word_alignment:
                        output_text = re.sub(pattern, "", supervisions[1].text)
                        output_text = re.sub(r'\s+', ' ', output_text).strip()
                        target_text = self.text_processor._process_example(context="", output=output_text)
                        # -1 to remove the eos token added by the text processor
                        target_text, target_text_length = torch.as_tensor(
                            target_text["answer_ids"][:-1]
                        ), torch.as_tensor(len(target_text["answer_ids"]) - 1)
                    else:
                        target_text, start_time_token, word_length = self._extract_text_and_time_tokens(text)
                        target_text_length = len(target_text)
                else:
                    raise Exception(f"Second speaker should be agent {cut}")

                instructions.append(instruction)
                instruction_lengths.append(instruction_length)
                target_texts.append(target_text)
                target_text_lengths.append(target_text_length)
                source_texts.append(source_text)
                source_text_lengths.append(source_text_length)
                text_start_time[-1].append(supervisions[1].start)
                text_end_time[-1].append(supervisions[1].start + supervisions[1].duration)
                text_start_time[-1][-1] = validate_time(text_start_time[-1][-1])
                text_end_time[-1][-1] = validate_time(text_end_time[-1][-1])
                if use_word_alignment:
                    word_lengths.append(word_length)
                    start_time_tokens.append(start_time_token)

        answer_audios, answer_audio_lens = None, None
        assert self.load_answer_audio
        assert not getattr(cut, "direct_s2s", False), "direct_s2s not supported when load_answer_audio is True"

        def load_audio_from_cut(cuts, name, sample_rate):
            answer_audio_lens = []
            answer_audios = []  # b*N
            features_lens = []
            for i, cut in enumerate(cuts):
                field = getattr(cut, name)
                if isinstance(field, list):
                    audios_list = field
                else:
                    audios_list = [field]
                for audios in audios_list:
                    if not isinstance(audios, Recording):
                        # TODO: tmp solution for multiturn
                        audios = Recording.from_file(audios['sources'][0]['source'])

                    answer_audio = torch.tensor(audios.resample(sample_rate).load_audio()).float()
                    answer_audio_len = torch.tensor(answer_audio.shape[1]).long()
                    answer_audios.append(answer_audio)
                    answer_audio_lens.append(answer_audio_len)
                    features_lens.append(
                        math.ceil(
                            answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                        )
                    )
            answer_audios = collate_vectors(
                [a.squeeze(0) for a in answer_audios], max_length=max(answer_audio_lens), padding_value=0.0
            ).float()
            answer_audio_lens = torch.tensor(answer_audio_lens).long()
            features_lens = torch.tensor(features_lens, dtype=torch.int)
            return answer_audios, answer_audio_lens, features_lens

        # in duplex data, user channel is kept in cut.recording and agent channel is kept in cut.target_audio
        # in the following, we keep target and source audio in different sample rates to be compatible with the single-turn and multi-turn branches
        # may not be necessary in future
        if hasattr(cuts[0], "target_audio"):
            # 22k target audio
            answer_audios, answer_audio_lens, features_lens = load_audio_from_cut(
                cuts, "target_audio", self.codec_sample_rate
            )
            # 16k source audio
            audio = [cut.resample(self.sample_rate).load_audio() for cut in cuts]
            audio_lens = [torch.tensor(a.shape[1]).long() for a in audio]
            audio = collate_vectors([a.squeeze(0) for a in audio], max_length=max(audio_lens), padding_value=0.0)
            audio_lens = torch.tensor(audio_lens).long()
        else:
            raise ValueError(
                "cut does not have target_audio. In duplex mode, recording keeps user channel and target_audio keeps agent channel"
            )

        text_pad_id = self.text_processor.pad_id

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        def get_step_by_time(text_start_time):
            text_start_step = (
                text_start_time
                * self.codec_sample_rate
                / self.codec_model_downsampling_factor
                // self.decoder_reduction_factor
            )
            return int(text_start_step) - 1

        cnt = 0
        skipped = 0
        new_target_texts = []
        new_source_texts = []
        answer_audios_first_turn = []
        answer_audios_first_turn_lens = []
        for i in range(len(num_turns)):
            each_target_texts = []
            total_steps = (
                torch.ceil(
                    answer_audio_lens[i] / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                ).int()
                + 1
            )
            cur_target_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )
            cur_source_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )

            # assert len(text_start_time[i]) == num_turns[i] // 2
            for j in range(num_turns[i] // 2):
                text_start_step = get_step_by_time(text_start_time[i][j])
                text_end_step = get_step_by_time(text_end_time[i][j]) + 1
                if text_end_step == total_steps:
                    text_end_step = total_steps - 1  # boundary case
                elif text_end_step > total_steps:
                    raise Exception("text_end_step too long")

                if text_start_step + 1 >= cur_target_text.shape[0] or text_end_step == text_start_step:
                    skipped += 1
                    continue  # the case of the last turn is user

                cur_target_text[text_start_step] = self.text_processor.bos_id
                cur_source_text[text_start_step] = self.text_processor.bos_id

                # if first turn
                if j == 0:
                    audio_start_step = int(
                        text_start_time[i][j] * self.codec_sample_rate
                    )  # multiply by sampling rate to convert seconds in frames
                    audio_end_step = int(
                        text_end_time[i][j] * self.codec_sample_rate
                    )  # multiply by sampling rate to convert seconds in frames
                    answer_audio_first_turn = answer_audios[i][audio_start_step:audio_end_step].unsqueeze(0)
                    answer_audios_first_turn.append(answer_audio_first_turn)
                    answer_audio_len = torch.tensor(answer_audio_first_turn.shape[1]).long()
                    answer_audios_first_turn_lens.append(answer_audio_len)

                if getattr(cut, "s2s_duplex", False):
                    # Note: text can be truncated
                    text_len = min(text_end_step - text_start_step - 1, target_texts[cnt].shape[0])
                    cur_target_text[(text_start_step + 1) : (text_start_step + 1 + text_len)] = target_texts[cnt][
                        :text_len
                    ]
                    src_text_len = min(text_end_step - text_start_step - 1, source_texts[cnt].shape[0])
                    cur_source_text[(text_start_step + 1) : (text_start_step + 1 + src_text_len)] = source_texts[cnt][
                        :src_text_len
                    ]

                elif getattr(cut, "s2s_duplex_align", False):
                    text_len_plus_eos = torch.tensor(text_end_step - text_start_step)
                    target_texts_expanded = self._expand_text_with_timestamps_and_word_lengths(
                        [target_texts[cnt]],
                        [word_lengths[cnt]],
                        [start_time_tokens[cnt]],
                        [text_len_plus_eos],
                        self.codec_model_downsampling_factor / self.codec_sample_rate,
                        pad_id=self.text_processor.unk_id,
                    )
                    cur_target_text[(text_start_step + 1) : (text_start_step + 1 + text_len_plus_eos)] = (
                        target_texts_expanded[0]
                    )
                else:
                    raise Exception("Undefined assistant channel text format.")

                cur_target_text[text_end_step] = self.text_processor.eos_id
                cur_source_text[text_end_step] = self.text_processor.eos_id
                cnt += 1

            new_target_texts.append(cur_target_text)
            new_source_texts.append(cur_source_text)

        target_texts_merge, target_text_lengths = collate_and_pad(new_target_texts)
        source_texts_merge, source_text_lengths = collate_and_pad(new_source_texts)

        answer_audios_first_turn = collate_vectors(
            [a.squeeze(0) for a in answer_audios_first_turn],
            max_length=max(answer_audios_first_turn_lens),
            padding_value=0.0,
        )
        answer_audios_first_turn_lens = torch.tensor(answer_audios_first_turn_lens).long()

        """
        import soundfile as sf
        sr = 16000
        file_name = "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/debug-samples/dataloader_youtube_input.wav"
        # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
        sf.write(file_name, audio[0].cpu().numpy().astype(np.float32), sr)

        sr = 22050
        file_name = "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/debug-samples/dataloader_youtube_target.wav"
        # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
        sf.write(file_name, answer_audios[0].cpu().numpy().astype(np.float32), sr)

        sr = 22050
        file_name = "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-full-duplex/debug-samples/dataloader_youtube_spk_ref.wav"
        # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
        sf.write(file_name, answer_audios_first_turn[0].cpu().numpy().astype(np.float32), sr)
        print(cuts[0])
        #print(list(cuts.ids)[0])
        """

        assert cnt + skipped == len(target_texts)
        assert target_texts_merge.shape[0] == len(num_turns)
        assert cnt + skipped + skipped_source == len(source_texts)
        assert len(target_texts) + len(source_texts) == sum(num_turns)
        assert source_texts_merge.shape[0] == len(num_turns)

        # note: the codec id in labels and contexts and others do not consider the offset e.g. speech_eos is 1002
        # the offset is all considered by SumVocabParallelEmbedding
        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,
            "audio_signal_length": audio_lens,
            "metadata": metadata,
            # For forward
            "instructions": None,
            "tokens": target_texts_merge,  # used in _reconfigure_and_process_inference_batch
            "target_texts_merge": target_texts_merge,  # used in prepare_llm_input
            "source_texts_merge": source_texts_merge,  # used in prepare_llm_input
            "contexts": target_texts_merge[:, :1],  # used in inference
            "context_lengths": torch.ones_like(target_text_lengths),  # placeholder
            "target_texts": target_texts_merge,
            "target_text_lengths": target_text_lengths,
            "answers": target_texts_merge,
            "answer_audio": answer_audios,
            "answer_audio_lens": answer_audio_lens,
            "num_turns": torch.Tensor(num_turns).long(),
            "speaker_ids": self.get_speaker_id(cuts),
            "answer_audios_first_turn": answer_audios_first_turn,
            "answer_audios_first_turn_lens": answer_audios_first_turn_lens,
        }

        if hasattr(cut, "include_sys"):  # assume no within batch mixing
            system_prompts_tensor = []
            system_prompts_tensor_length = []
            for cut in cuts:

                source_text = self.text_processor._process_example(context="", output=cut.system_prompt)
                # -1 to remove the eos token added by the text processor
                source_text, source_text_length = torch.as_tensor(source_text["answer_ids"][:-1]), torch.as_tensor(
                    len(source_text["answer_ids"]) - 1
                )
                system_prompts_tensor.append(source_text)
                system_prompts_tensor_length.append(source_text_length)
            system_prompts_tensor = collate_vectors(
                system_prompts_tensor,
                max_length=max(system_prompts_tensor_length),
                padding_value=self.text_processor.eos_id,
            ).long()
            system_prompts_tensor_length = torch.tensor(system_prompts_tensor_length).long()
            return_batch['system_prompts'] = system_prompts_tensor
            return_batch['system_prompts_length'] = system_prompts_tensor_length
            return_batch['contexts'] = system_prompts_tensor
            return_batch['context_lengths'] = system_prompts_tensor_length

        return return_batch

    def get_speaker_id(self, cuts):
        speaker_ids = [getattr(cut, "speaker_id", 0) for cut in cuts]
        return torch.tensor(speaker_ids).long()

    def __getitem__duplex_overlap_(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        import re

        cuts = cuts.sort_by_duration()

        answer_audios, answer_audio_lens = None, None
        assert self.load_answer_audio

        def load_audio_from_cut(cuts, name, sample_rate):
            answer_audio_lens = []
            answer_audios = []  # b*N
            features_lens = []
            for i, cut in enumerate(cuts):
                field = getattr(cut, name)
                if isinstance(field, list):
                    audios_list = field
                else:
                    audios_list = [field]
                for audios in audios_list:
                    if not isinstance(audios, Recording):
                        # TODO: tmp solution for multiturn
                        audios = Recording.from_file(audios['sources'][0]['source'])

                    answer_audio = torch.tensor(audios.resample(sample_rate).load_audio()).float()
                    answer_audio_len = torch.tensor(answer_audio.shape[1]).long()
                    answer_audios.append(answer_audio)
                    answer_audio_lens.append(answer_audio_len)
                    features_lens.append(
                        math.ceil(
                            answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                        )
                    )
            answer_audios = collate_vectors(
                [a.squeeze(0) for a in answer_audios], max_length=max(answer_audio_lens), padding_value=0.0
            ).float()
            answer_audio_lens = torch.tensor(answer_audio_lens).long()
            features_lens = torch.tensor(features_lens, dtype=torch.int)
            return answer_audios, answer_audio_lens, features_lens

        # in duplex data, user channel is kept in cut.recording and agent channel is kept in cut.target_audio
        # in the following, we keep target and source audio in different sample rates to be compatible with the single-turn and multi-turn branches
        # may not be necessary in future
        if hasattr(cuts[0], "target_audio"):
            # 22k target audio
            answer_audios, answer_audio_lens, features_lens = load_audio_from_cut(
                cuts, "target_audio", self.codec_sample_rate
            )
            # 16k source audio
            audio = [cut.resample(self.sample_rate).load_audio() for cut in cuts]
            audio_lens = [torch.tensor(a.shape[1]).long() for a in audio]
            audio = collate_vectors([a.squeeze(0) for a in audio], max_length=max(audio_lens), padding_value=0.0)
            audio_lens = torch.tensor(audio_lens).long()
        else:
            raise ValueError(
                "cut does not have target_audio. In duplex mode, recording keeps user channel and target_audio keeps agent channel"
            )

        text_pad_id = self.text_processor.pad_id

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        def get_step_by_time(text_start_time):
            text_start_step = (
                text_start_time
                * self.codec_sample_rate
                / self.codec_model_downsampling_factor
                // self.decoder_reduction_factor
            )
            return int(text_start_step) - 1

        metadata = []
        target_texts, target_text_lengths = [], []
        source_texts, source_text_lengths = [], []
        num_turns = []
        new_target_texts = []
        new_source_texts = []
        answer_audios_first_turn = []
        answer_audios_first_turn_lens = []
        new_agent_turns = []
        for id, cut in enumerate(cuts):

            def validate_time(input_time):
                if input_time > cut.duration + 0.16:
                    logging.info(f"{input_time} > {cut.duration} in {cut}")
                return min(input_time, cut.duration)

            if not hasattr(cut, 'user_segments'):
                raise ValueError(f"user_segments: {cut}")

            num_turns.append(len(cut.user_segments) + len(cut.agent_segments))
            metadata.append({'audio_filepath': cut.id + '.wav'})
            total_steps = (
                torch.ceil(
                    answer_audio_lens[id] / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                ).int()
                + 1
            )

            def get_text_from_segments(segments, total_steps):
                cur_target_text = torch.full(
                    [total_steps],
                    (
                        self.text_processor.tokenizer.pad_id
                        if hasattr(self.text_processor.tokenizer, 'pad_id')
                        and self.text_processor.tokenizer.pad_id >= 0
                        else self.text_processor.tokenizer.unk_id
                    ),
                )
                cur_agent_turn = torch.full(
                    [total_steps],
                    (0),
                )
                for i, segment in enumerate(segments):
                    # Extract agent text
                    pattern = r"<\|\d+\|>"
                    output_text = re.sub(pattern, "", segment['text'])
                    output_text = re.sub(r'\s+', ' ', output_text).strip()
                    target_text = self.text_processor._process_example(context="", output=output_text)
                    # -1 to remove the eos token added by the text processor
                    target_text, target_text_length = torch.as_tensor(target_text["answer_ids"][:-1]), torch.as_tensor(
                        len(target_text["answer_ids"]) - 1
                    )
                    target_texts.append(target_text)
                    target_text_lengths.append(target_text_length)
                    text_start_time = segment['start']
                    text_end_time = segment['end']
                    text_start_time = validate_time(text_start_time)
                    text_end_time = validate_time(text_end_time)
                    text_start_step = get_step_by_time(text_start_time)
                    text_end_step = get_step_by_time(text_end_time) + 1
                    if text_end_step == total_steps:
                        text_end_step = total_steps - 1  # boundary case
                    elif text_end_step > total_steps:
                        raise Exception("text_end_step too long")

                    cur_target_text[text_start_step] = self.text_processor.bos_id
                    # Note: text can be truncated
                    text_len = min(text_end_step - text_start_step - 1, target_text.shape[0])
                    cur_target_text[(text_start_step + 1) : (text_start_step + 1 + text_len)] = target_text[:text_len]
                    cur_target_text[text_end_step] = self.text_processor.eos_id
                    cur_agent_turn[text_start_step : (text_end_step + 1)] = 1
                return cur_target_text, cur_agent_turn

            def get_answer_audio_first_turn_from_segments(segments, answer_audios, id):
                # consider first segment
                segment = segments[0]
                text_start_time = validate_time(segment['start'])
                text_end_time = validate_time(segment['end'])
                audio_start_step = int(
                    validate_time(segment['start']) * self.codec_sample_rate
                )  # multiply by sampling rate to convert seconds in frames
                audio_end_step = int(
                    validate_time(segment['end']) * self.codec_sample_rate
                )  # multiply by sampling rate to convert seconds in frames
                answer_audio_first_turn = answer_audios[id][audio_start_step:audio_end_step].unsqueeze(0)
                answer_audio_len = torch.tensor(answer_audio_first_turn.shape[1]).long()
                return answer_audio_first_turn, answer_audio_len

            cur_target_text, cur_agent_turn = get_text_from_segments(cut.agent_segments, total_steps)
            cur_source_text, _ = get_text_from_segments(cut.user_segments, total_steps)

            new_target_texts.append(cur_target_text)
            new_source_texts.append(cur_source_text)
            new_agent_turns.append(cur_agent_turn)

            # get first turn answer audio
            answer_audio_first_turn, answer_audio_first_turn_len = get_answer_audio_first_turn_from_segments(
                cut.agent_segments, answer_audios, id
            )
            answer_audios_first_turn.append(answer_audio_first_turn)
            answer_audios_first_turn_lens.append(answer_audio_first_turn_len)

        target_texts_merge, target_text_lengths = collate_and_pad(new_target_texts)
        source_texts_merge, source_text_lengths = collate_and_pad(new_source_texts)
        agent_turns_merge, agent_turns_lengths = collate_and_pad(new_agent_turns)
        answer_audios_first_turn = collate_vectors(
            [a.squeeze(0) for a in answer_audios_first_turn],
            max_length=max(answer_audios_first_turn_lens),
            padding_value=0.0,
        )
        answer_audios_first_turn_lens = torch.tensor(answer_audios_first_turn_lens).long()

        assert target_texts_merge.shape[0] == len(num_turns)

        # note: the codec id in labels and contexts and others do not consider the offset e.g. speech_eos is 1002
        # the offset is all considered by SumVocabParallelEmbedding
        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,
            "audio_signal_length": audio_lens,
            "metadata": metadata,
            # For forward
            "instructions": None,
            "tokens": target_texts_merge,  # used in _reconfigure_and_process_inference_batch
            "target_texts_merge": target_texts_merge,  # used in prepare_llm_input
            "source_texts_merge": source_texts_merge,  # used in prepare_llm_input
            "agent_turns_merge": agent_turns_merge,
            "contexts": target_texts_merge[:, :1],  # used in inference
            "context_lengths": torch.ones_like(target_text_lengths),
            "target_texts": target_texts_merge,
            "target_text_lengths": target_text_lengths,
            "source_text_lengths": source_text_lengths,
            "answers": target_texts_merge,
            "answer_audio": answer_audios,
            "answer_audio_lens": answer_audio_lens,
            "num_turns": torch.Tensor(num_turns).long(),
            "s2s_duplex_overlap": torch.ones_like(target_text_lengths),
            "speaker_ids": self.get_speaker_id(cuts),
            "answer_audios_first_turn": answer_audios_first_turn,
            "answer_audios_first_turn_lens": answer_audios_first_turn_lens,
        }

        return return_batch

    def __getitem__tts_(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        import re

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        # adapted from https://github.com/blisc/NeMo/blob/magpietts_2503/nemo/collections/tts/data/text_to_speech_dataset_lhotse.py
        cuts = cuts.sort_by_duration()
        answer_audios = []
        answer_audio_lens = []
        features_lens = []
        target_texts = []
        target_text_lengths = []
        answer_audios_first_turn = []
        answer_audios_first_turn_lens = []
        metadata = []
        new_target_texts = []
        new_source_texts = []
        num_turns = []
        text_pad_id = self.text_processor.pad_id
        for i, cut in enumerate(cuts):
            # load target/answer audio
            answer_audio = torch.FloatTensor(cut.target_audio.resample(self.codec_sample_rate).load_audio())
            # add 2-3 tokens equivalent silence and set text_start_step and text_end_step accordently
            random_silence_tokens_left = random.randint(2, 3)
            random_silence_tokens_right = random.randint(2, 3)
            r_padding = torch.zeros(1, random_silence_tokens_right * self.codec_model_downsampling_factor)
            l_padding = torch.zeros(1, random_silence_tokens_left * self.codec_model_downsampling_factor)
            answer_audio = torch.cat([l_padding, answer_audio, r_padding], dim=1)

            answer_audio_len = torch.tensor(answer_audio.shape[1]).long()
            answer_audios.append(answer_audio)
            answer_audio_lens.append(answer_audio_len)
            features_lens.append(
                math.ceil(answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor)
            )

            answer_audios_first_turn.append(answer_audio)
            answer_audios_first_turn_lens.append(answer_audio_len)
            metadata.append({'audio_filepath': cut.id + '.wav'})
            num_turns.append(1)  # num turns for tts data is always 1

            # text_start_step is random_silence_tokens_left - 1, to include the addition of bos token and extra -1 because it start on zero
            text_start_step = random_silence_tokens_left - 2
            # text_end_step is total audio len divided by self.codec_model_downsampling_factor / self.decoder_reduction_factor to get frame level len minus  random_silence_tokens_right - 2
            text_end_step = int(
                answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
            ) - (random_silence_tokens_right - 2)

            total_steps = (
                torch.ceil(
                    answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                ).int()
                + 1
            )

            cur_target_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )
            # create emptly sorce text
            cur_source_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )

            if cut.supervisions[1].speaker == "agent":
                output_text = cut.supervisions[1].text
                # Extract agent text
                pattern = r"<\|\d+\|>"
                output_text = re.sub(pattern, "", output_text)
                output_text = re.sub(r'\s+', ' ', output_text).strip()
                target_text = self.text_processor._process_example(context="", output=output_text)
                # -1 to remove the eos token added by the text processor
                target_text, target_text_length = torch.as_tensor(target_text["answer_ids"][:-1]), torch.as_tensor(
                    len(target_text["answer_ids"]) - 1
                )
                target_texts.append(target_text)
                target_text_lengths.append(target_text_length)

                if text_end_step == total_steps:
                    text_end_step = total_steps - 1  # boundary case

                cur_target_text[text_start_step] = self.text_processor.bos_id
                # Note: text can be truncated
                text_len = min(text_end_step - text_start_step - 1, target_text.shape[0])
                cur_target_text[(text_start_step + 1) : (text_start_step + 1 + text_len)] = target_text[:text_len]
                cur_target_text[text_end_step] = self.text_processor.eos_id

                # for source text the begging and end are oposite, so add eos when it was bos and eos and it was bos
                cur_source_text[max(text_end_step - 1, 0)] = (
                    self.text_processor.eos_id
                )  # -1 to make sure that both eos
                cur_source_text[min(text_end_step + 1, cur_source_text.size(0) - 1)] = (
                    self.text_processor.bos_id
                )  # +1 to make sure that both eos bos are in the tensor if we use both channels

                new_target_texts.append(cur_target_text)
                new_source_texts.append(cur_source_text)

        # collate and pad target texts
        target_texts_merge, target_text_lengths = collate_and_pad(new_target_texts)

        # collate and pad source dummy texts
        source_texts_merge, source_text_lengths = collate_and_pad(new_source_texts)

        # collate_vectors answer_audios
        answer_audios = collate_vectors(
            [a.squeeze(0) for a in answer_audios], max_length=max(answer_audio_lens), padding_value=0.0
        )
        answer_audio_lens = torch.tensor(answer_audio_lens).long()

        # prepare answer_audios_first_turn that will be used for speaker conditioning
        answer_audios_first_turn = collate_vectors(
            [a.squeeze(0) for a in answer_audios_first_turn],
            max_length=max(answer_audios_first_turn_lens),
            padding_value=0.0,
        )
        answer_audios_first_turn_lens = torch.tensor(answer_audios_first_turn_lens).long()

        # make features_lens a tensor
        features_lens = torch.tensor(features_lens, dtype=torch.int)

        # user audio is a zero tensor as answer_audios downsampled to 16khz
        audio = torch.zeros(answer_audios.size(), dtype=answer_audios.dtype)  # .uniform_(-0.1,0.1)
        audio = torchaudio.functional.resample(audio, self.codec_sample_rate, self.sample_rate)
        audio_lens = (answer_audio_lens * (self.sample_rate / self.codec_sample_rate)).long()

        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,  # audio signal is a zero like
            "audio_signal_length": audio_lens,
            "metadata": metadata,
            # For forward
            "instructions": None,
            "tokens": target_texts_merge,  # used in _reconfigure_and_process_inference_batch
            "target_texts_merge": target_texts_merge,  # used in prepare_llm_input
            "source_texts_merge": source_texts_merge,  # used in prepare_llm_input
            "contexts": target_texts_merge[:, :1],  # used in inference
            "context_lengths": torch.ones_like(target_text_lengths),
            "target_texts": target_texts_merge,
            "target_text_lengths": target_text_lengths,
            "source_text_lengths": source_text_lengths,
            "answers": target_texts_merge,
            "answer_audio": answer_audios,
            "answer_audio_lens": answer_audio_lens,
            "num_turns": torch.Tensor(num_turns).long(),
            "speaker_ids": self.get_speaker_id(cuts),
            "answer_audios_first_turn": answer_audios_first_turn,
            "answer_audios_first_turn_lens": answer_audios_first_turn_lens,
        }

        return return_batch

    def __getitem__tts_repeat_after_me(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        import re

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        # adapted from https://github.com/blisc/NeMo/blob/magpietts_2503/nemo/collections/tts/data/text_to_speech_dataset_lhotse.py
        cuts = cuts.sort_by_duration()
        user_audios = []
        user_audios_lens = []
        answer_audios = []
        answer_audio_lens = []
        features_lens = []
        target_texts = []
        target_text_lengths = []
        answer_audios_first_turn = []
        answer_audios_first_turn_lens = []
        metadata = []
        new_target_texts = []
        new_source_texts = []
        num_turns = []
        text_pad_id = self.text_processor.pad_id
        for i, cut in enumerate(cuts):
            # load target/answer audio
            cur_answer_audio = torch.FloatTensor(cut.target_audio.resample(self.codec_sample_rate).load_audio())
            # convert answer to the input sr
            cur_answer_audio_input_sr = torchaudio.functional.resample(
                cur_answer_audio, self.codec_sample_rate, self.sample_rate
            )

            # define silences between turns
            turn_silence_sec = 0.32
            silence_padding_input = torch.zeros((1, int(turn_silence_sec * self.sample_rate)))  # input sr silence
            silence_padding_output = torch.zeros(
                (1, int(turn_silence_sec * self.codec_sample_rate))
            )  # output sr silence

            # generate target audio with end 32 ms padding
            cur_answer_audio_with_padding = torch.cat([cur_answer_audio, silence_padding_output], dim=1)
            # generates user audio with prompt and padding
            user_audio_with_prompt_and_padding = torch.cat(
                [self.prompt_audio, silence_padding_input, cur_answer_audio_input_sr, silence_padding_input], dim=1
            )

            # downsample it to the input/output sr
            cur_answer_audio_with_padding_input_sr = torchaudio.functional.resample(
                cur_answer_audio_with_padding, self.codec_sample_rate, self.sample_rate
            )
            user_audio_with_prompt_and_padding_output_sr = torchaudio.functional.resample(
                user_audio_with_prompt_and_padding, self.sample_rate, self.codec_sample_rate
            )

            # make the final user and answer audios
            user_audio = torch.cat(
                [user_audio_with_prompt_and_padding, torch.zeros_like(cur_answer_audio_with_padding_input_sr)], dim=1
            )
            answer_audio = torch.cat(
                [torch.zeros_like(user_audio_with_prompt_and_padding_output_sr), cur_answer_audio_with_padding], dim=1
            )

            # add user and answer audios to the lists
            user_audios.append(user_audio)
            user_audio_len = torch.tensor(user_audio.shape[1]).long()
            user_audios_lens.append(user_audio_len)
            answer_audios.append(answer_audio)
            answer_audio_len = torch.tensor(answer_audio.shape[1]).long()
            answer_audio_lens.append(answer_audio_len)
            features_lens.append(
                math.ceil(answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor)
            )

            answer_audios_first_turn.append(cur_answer_audio)
            answer_audios_first_turn_lens.append(torch.tensor(cur_answer_audio.shape[1]).long())
            metadata.append({'audio_filepath': cut.id + '.wav'})
            num_turns.append(1)  # num turns for tts data is always 1

            # target text_start_step is the size of user_audio_with_prompt_and_padding minus 1 to avoid cut real speech
            text_start_step = (
                int(
                    user_audio_with_prompt_and_padding_output_sr.size(1)
                    / self.codec_model_downsampling_factor
                    / self.decoder_reduction_factor
                )
                - 1
            )
            # target text_end_step is total audio len divided by self.codec_model_downsampling_factor / self.decoder_reduction_factor to get frame level len minus 1 to avoid remove real speech
            text_end_step = (
                int(answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor) - 1
            )

            total_steps = (
                torch.ceil(
                    answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                ).int()
                + 1
            )

            cur_target_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )
            # create emptly sorce text
            cur_source_text = torch.full(
                [total_steps],
                (
                    self.text_processor.tokenizer.pad_id
                    if hasattr(self.text_processor.tokenizer, 'pad_id') and self.text_processor.tokenizer.pad_id >= 0
                    else self.text_processor.tokenizer.unk_id
                ),
            )

            if cut.supervisions[1].speaker == "agent":
                output_text = cut.supervisions[1].text
                # Extract agent text
                pattern = r"<\|\d+\|>"
                output_text = re.sub(pattern, "", output_text)
                output_text = re.sub(r'\s+', ' ', output_text).strip()
                target_text = self.text_processor._process_example(context="", output=output_text)

                source_text = "Can you repeat after me? " + output_text
                source_text = self.text_processor._process_example(context="", output=source_text)

                # -1 to remove the eos token added by the text processor
                target_text, target_text_length = torch.as_tensor(target_text["answer_ids"][:-1]), torch.as_tensor(
                    len(target_text["answer_ids"]) - 1
                )
                source_text, source_text_length = torch.as_tensor(source_text["answer_ids"][:-1]), torch.as_tensor(
                    len(source_text["answer_ids"]) - 1
                )

                target_texts.append(target_text)
                target_text_lengths.append(target_text_length)

                if text_end_step == total_steps:
                    text_end_step = total_steps - 1  # boundary case

                cur_target_text[text_start_step] = self.text_processor.bos_id
                text_len = min(text_end_step - text_start_step - 1, target_text.shape[0])
                cur_target_text[(text_start_step + 1) : (text_start_step + 1 + text_len)] = target_text[:text_len]
                cur_target_text[text_end_step] = self.text_processor.eos_id

                # for source text, the bos happens in the position and eos happens right before target text_start_step
                source_text_start_step = 0
                source_text_end_step = text_start_step - 1

                cur_source_text[source_text_start_step] = self.text_processor.bos_id
                cur_source_text[source_text_end_step] = self.text_processor.eos_id
                text_len = min(source_text_end_step - source_text_start_step - 1, source_text.shape[0])
                cur_source_text[(source_text_start_step + 1) : (source_text_start_step + 1 + text_len)] = source_text[
                    :text_len
                ]

                new_target_texts.append(cur_target_text)
                new_source_texts.append(cur_source_text)

        # collate and pad target texts
        target_texts_merge, target_text_lengths = collate_and_pad(new_target_texts)

        # collate and pad source dummy texts
        source_texts_merge, source_text_lengths = collate_and_pad(new_source_texts)

        # collate_vectors answer_audios
        answer_audios = collate_vectors(
            [a.squeeze(0) for a in answer_audios], max_length=max(answer_audio_lens), padding_value=0.0
        )
        answer_audio_lens = torch.tensor(answer_audio_lens).long()

        # collate_vectors user_audios
        audio = collate_vectors(
            [a.squeeze(0) for a in user_audios], max_length=max(user_audios_lens), padding_value=0.0
        )
        audio_lens = torch.tensor(user_audios_lens).long()

        # prepare answer_audios_first_turn that will be used for speaker conditioning
        answer_audios_first_turn = collate_vectors(
            [a.squeeze(0) for a in answer_audios_first_turn],
            max_length=max(answer_audios_first_turn_lens),
            padding_value=0.0,
        )
        answer_audios_first_turn_lens = torch.tensor(answer_audios_first_turn_lens).long()

        # make features_lens a tensor
        features_lens = torch.tensor(features_lens, dtype=torch.int)

        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,  # audio signal is a zero like
            "audio_signal_length": audio_lens,
            "metadata": metadata,
            # For forward
            "instructions": None,
            "tokens": target_texts_merge,  # used in _reconfigure_and_process_inference_batch
            "target_texts_merge": target_texts_merge,  # used in prepare_llm_input
            "source_texts_merge": source_texts_merge,  # used in prepare_llm_input
            "contexts": target_texts_merge[:, :1],  # used in inference
            "context_lengths": torch.ones_like(target_text_lengths),
            "target_texts": target_texts_merge,
            "target_text_lengths": target_text_lengths,
            "source_text_lengths": source_text_lengths,
            "answers": target_texts_merge,
            "answer_audio": answer_audios,
            "answer_audio_lens": answer_audio_lens,
            "num_turns": torch.Tensor(num_turns).long(),
            "speaker_ids": self.get_speaker_id(cuts),
            "answer_audios_first_turn": answer_audios_first_turn,
            "answer_audios_first_turn_lens": answer_audios_first_turn_lens,
        }

        return return_batch

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        import re

        # full text data goes here
        from nemo.collections.common.data.lhotse.text_adapters import (
            NeMoMultimodalConversation,
            NeMoSFTExample,
            SourceTargetTextExample,
        )

        text_examples = cuts.filter(
            lambda c: isinstance(c, (SourceTargetTextExample, NeMoSFTExample, NeMoMultimodalConversation))
        )
        if text_examples:
            # reformat text for duplex format
            if getattr(text_examples[0], "s2s_duplex", False):
                use_random_padding = getattr(text_examples[0], "random_padding_vtblender", False)  # 2.5 to 4
                limit_max_seq_length = getattr(text_examples[0], "limit_max_seq_length", False)
                pad_id = self.text_processor.pad_id
                bos_id = self.text_processor.bos_id
                eos_id = self.text_processor.eos_id

                input_ids_all = []
                answer_ids_all = []
                context_ids_all = []
                answer_masks_all = []
                for text_example in text_examples:
                    input_ids_list = []
                    answer_ids_list = []
                    context_ids_list = []
                    current_sample_len = 0
                    for turn in text_example.turns:
                        cur_turn_tokens = self.text_processor._process_example(context="", output=turn.value)[
                            "answer_ids"
                        ][
                            :-1
                        ]  # -1 to remove the eos token added by the text processor

                        if not use_random_padding:
                            pad_full_cur_input = np.full(shape=len(cur_turn_tokens) + 3, fill_value=pad_id)
                            # create a copy to fill the input
                            cur_input_text = np.copy(pad_full_cur_input)

                            cur_input_text[0] = bos_id
                            cur_input_text[-2] = eos_id
                            cur_input_text[1:-2] = cur_turn_tokens
                            # keep last token as pad, as done one speech mode
                        else:
                            # add extra padding to the text only channel input to emulate speech
                            number_tokens_with_random_padding = int(len(cur_turn_tokens) * random.uniform(1.7, 2.1))
                            pad_full_cur_input = np.full(
                                shape=number_tokens_with_random_padding + 3, fill_value=pad_id
                            )
                            # create a copy to fill the input
                            cur_input_text = np.copy(pad_full_cur_input)

                            cur_input_text[0] = bos_id
                            cur_input_text[-2] = eos_id
                            cur_input_text[1 : len(cur_turn_tokens) + 1] = cur_turn_tokens

                        current_sample_len += len(cur_turn_tokens)
                        # if it has already at least a full first turn conversation and reaches the limit_max_seq_length, ignore the rest
                        if (
                            limit_max_seq_length
                            and len(input_ids_list) >= 2
                            and (current_sample_len > limit_max_seq_length)
                        ):
                            continue

                        if turn.role == "user":
                            input_ids_list.append(cur_input_text)
                            answer_ids_list.append(pad_full_cur_input)
                        else:
                            input_ids_list.append(pad_full_cur_input)
                            answer_ids_list.append(cur_input_text)

                        context_ids_list.append(cur_turn_tokens)

                    input_ids = np.concatenate(input_ids_list, axis=0)
                    input_ids_all.append(input_ids)
                    answer_ids = np.concatenate(answer_ids_list, axis=0)
                    answer_ids_all.append(answer_ids)
                    context_ids = np.concatenate(context_ids_list, axis=0)
                    context_ids_all.append(context_ids)
                    # add mask based on the whole output, considering the padding as done on speech duplex mode
                    answer_masks_all.append(np.ones(len(answer_ids)))

                text_minibatch = dict(
                    text_input_ids=collate_vectors_lhotse(input_ids_all, padding_value=pad_id),
                    text_labels_ids=collate_vectors_lhotse(answer_ids_all, padding_value=pad_id),
                    text_context_ids=collate_vectors_lhotse(context_ids_all, padding_value=pad_id),
                    text_loss_masks=collate_vectors_lhotse(answer_masks_all, padding_value=0),
                )
            else:
                pad_id = self.text_processor.pad_id
                text_minibatch = dict(
                    text_input_ids=collate_vectors_lhotse([e.input_ids for e in text_examples], padding_value=pad_id),
                    text_answer_ids=collate_vectors_lhotse(
                        [e.answer_ids for e in text_examples], padding_value=pad_id
                    ),
                    text_context_ids=collate_vectors_lhotse(
                        [e.context_ids for e in text_examples], padding_value=pad_id
                    ),
                    text_loss_masks=collate_vectors_lhotse([e.mask for e in text_examples], padding_value=0),
                )
                text_minibatch["text_labels_ids"] = text_minibatch["text_input_ids"][:, 1:]
                text_minibatch["text_input_ids"] = text_minibatch["text_input_ids"][:, :-1]
                text_minibatch["text_loss_masks"] = text_minibatch["text_loss_masks"][:, 1:]

            return text_minibatch

        # full-duplex data goes here
        if getattr(cuts[0], "s2s_duplex", False) or getattr(cuts[0], "s2s_duplex_align", False):
            return self.__getitem__duplex_(cuts)
        if getattr(cuts[0], "s2s_duplex_overlap", False):
            return self.__getitem__duplex_overlap_(cuts)
        if getattr(cuts[0], "tts", False):
            return self.__getitem__tts_(cuts)
        if getattr(cuts[0], "tts_repeat_after_me", False):
            return self.__getitem__tts_repeat_after_me(cuts)
        '''
        # half-duplex single turn s2s data and multi turn s2s data go here
        # TODO: the following stanza can be removed or cleaned if we only need duplex s2s
        '''

        cuts = cuts.sort_by_duration()

        metadata = []
        instructions, instruction_lengths = [], []
        target_texts, target_text_lengths = [], []
        remove_ids = []
        start_time_tokens, word_lengths = [], []
        num_turns = []
        for id, cut in enumerate(cuts):
            num_turns.append(len(cut.supervisions))
            metadata.append({'audio_filepath': cut.id + '.wav'})
            # treat multiturn data as multiple batch each with 2-turn conversation
            for i in range(0, len(cut.supervisions), 2):
                supervisions = cut.supervisions[i : i + 2]
                # TODO: the following use of _process_example is not ideal. Should update
                if supervisions[0].speaker != "user":
                    logging.info(f"First speaker should be user {cut}")
                instruction = self.text_processor._process_example(context=supervisions[0].text, output="")
                instruction, instruction_length = torch.as_tensor(instruction["input_ids"][:-1]), torch.as_tensor(
                    len(instruction["input_ids"]) - 1
                )

                if supervisions[1].speaker == "agent":
                    use_timestamp = getattr(cut, "s2s_align", False)
                    text = supervisions[1].text
                    if not use_timestamp:
                        pattern = r"<\|\d+\|>"
                        output_text = re.sub(pattern, "", text)
                        output_text = re.sub(r'\s+', ' ', output_text).strip()
                        target_text = self.text_processor._process_example(context="", output=output_text)
                        # -1 to remove the eos token added by the text processor
                        target_text, target_text_length = torch.as_tensor(
                            target_text["answer_ids"][:-1]
                        ), torch.as_tensor(len(target_text["answer_ids"]) - 1)
                    else:
                        target_text, start_time_token, word_length = self.extract_text_and_time_tokens(text)
                        target_text_length = len(target_text)
                else:
                    raise Exception("Second speaker should be agent")

                instructions.append(instruction)
                instruction_lengths.append(instruction_length)
                target_texts.append(target_text)
                target_text_lengths.append(target_text_length)
                if use_timestamp:
                    word_lengths.append(word_length)
                    start_time_tokens.append(start_time_token)

        answer_audios, answer_audio_lens = None, None
        assert self.load_answer_audio
        assert not getattr(cut, "direct_s2s", False), "direct_s2s not supported when load_answer_audio is True"

        # TODO(subhankarg) load answer audio from cut.target_codes logic
        def load_audio_from_cut(cuts, name, sample_rate):
            answer_audio_lens = []
            answer_audios = []  # b*N
            features_lens = []
            for i, cut in enumerate(cuts):
                field = getattr(cut, name)
                if isinstance(field, list):
                    audios_list = field
                else:
                    audios_list = [field]
                assert num_turns[i] / 2 == len(audios_list)
                for audios in audios_list:
                    from lhotse import Recording

                    if not isinstance(audios, Recording):
                        # TODO: tmp solution for multiturn
                        audios = Recording.from_file(audios['sources'][0]['source'])

                    answer_audio = torch.tensor(audios.resample(sample_rate).load_audio()).float()
                    answer_audio_len = torch.tensor(answer_audio.shape[1]).long()
                    answer_audios.append(answer_audio)
                    answer_audio_lens.append(answer_audio_len)
                    features_lens.append(
                        math.ceil(
                            answer_audio_len / self.codec_model_downsampling_factor / self.decoder_reduction_factor
                        )
                    )
            answer_audios = collate_vectors(
                [a.squeeze(0) for a in answer_audios], max_length=max(answer_audio_lens), padding_value=0.0
            ).float()
            answer_audio_lens = torch.tensor(answer_audio_lens).long()
            # Prepare dummy target_codec with speech_pad_id and eos_tensor, the dummy values will be filled in training_step or validation_step
            # once the audio codecs are extracted from the audio.
            features_lens = torch.tensor(features_lens, dtype=torch.int)
            return answer_audios, answer_audio_lens, features_lens

        # treat multiturn data as multiple batch each with 2-turn conversation
        if hasattr(cuts[0], "target_audios"):  # multi-turn
            all_target_audios, all_target_audio_lens, target_features_lens = load_audio_from_cut(
                cuts, "target_audios", self.codec_sample_rate
            )
            assert hasattr(cuts[0], "source_audios")
            all_source_audios, all_source_audio_lens, source_features_lens = load_audio_from_cut(
                cuts, "source_audios", self.sample_rate
            )
            assert all_target_audios.shape[0] == all_source_audios.shape[0]
            assert all_target_audios.shape[0] == len(instructions)
            assert all_target_audios.shape[0] == len(target_texts)
            answer_audios = all_target_audios
            answer_audio_lens = all_target_audio_lens
            audio = all_source_audios
            audio_lens = all_source_audio_lens
            features_lens = target_features_lens
        elif hasattr(cuts[0], "target_audio"):  # single-turn
            # 22k target audio
            answer_audios, answer_audio_lens, features_lens = load_audio_from_cut(
                cuts, "target_audio", self.codec_sample_rate
            )
            # 16k source audio
            audio = [cut.resample(self.sample_rate).load_audio() for cut in cuts]
            audio_lens = [torch.tensor(a.shape[1]).long() for a in audio]
            # Resample audio waveform here since cuts.resample causes core dump sometimes
            # cuts_sample_rates = [c.recording.sampling_rate for c in cuts]
            # import torchaudio
            # audio = [torchaudio.functional.resample(a, orig_sample_rate, self.sample_rate).squeeze(0) for a, orig_sample_rate in zip(audio, cuts_sample_rates)]
            # audio_lens = (torch.IntTensor(audio_lens) * (self.sample_rate / torch.IntTensor(cuts_sample_rates))).int()
            audio = collate_vectors([a.squeeze(0) for a in audio], max_length=max(audio_lens), padding_value=0.0)
            audio_lens = torch.tensor(audio_lens).long()
        else:
            raise ValueError("cut does not have target_audio or target_audios")

        return self.form_2turn_batch(
            cuts,
            features_lens,
            target_texts,
            instructions,
            audio,
            audio_lens,
            answer_audios,
            answer_audio_lens,
            metadata,
            word_lengths,
            start_time_tokens,
            num_turns,  # used to recover multi-turn format in modeling code
        )

    # TODO: the following stanza can be removed or cleaned if we only need duplex s2s
    def form_2turn_batch(
        self,
        cuts,
        features_lens,
        target_texts,
        instructions,
        audio,
        audio_lens,
        answer_audios,
        answer_audio_lens,
        metadata,
        word_lengths,
        start_time_tokens,
        num_turns,
    ):

        text_pad_id = self.text_processor.pad_id
        text_unk_id = self.text_processor.unk_id
        text_bos_id = self.text_processor.bos_id
        text_eos_id = self.text_processor.eos_id

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        def _convert_text_to_3d_tensor(texts, include_eos=True, tokens_to_generate=0):
            texts, text_lengths = collate_and_pad(texts)
            texts_expanded = get_3d_empty_tensor(
                texts.shape[0], texts.shape[1] + 1 + tokens_to_generate, text_pad_id, self.speech_pad_id
            )
            for i, text_length in enumerate(text_lengths):
                texts_expanded[i, :text_length, 0] = texts[i, :text_length]
                texts_expanded[i, :text_length, 1:] = self.speech_unk_id
                eos_tensor = torch.full(
                    (1, self.n_speech_codebooks * self.decoder_reduction_factor + 1), self.speech_bos_id
                ).to(torch.int)
                eos_tensor[:, 0] = self.text_processor.eos_id

                texts_expanded[i, text_length, :] = eos_tensor
            if not include_eos:
                texts_expanded = texts_expanded[:, :-1]
            return texts, text_lengths, texts_expanded

        batch_size = audio.shape[0]
        # TODO: can remove the following except features_lens
        target_codec = get_3d_empty_tensor(batch_size, max(features_lens).item() + 1, text_pad_id, self.speech_pad_id)
        eos_tensor = torch.full((1, target_codec.shape[-1]), self.speech_eos_id).to(torch.int)
        eos_tensor[:, 0] = self.text_processor.unk_id
        for i in range(batch_size):
            target_codec[i, : features_lens[i], 0] = text_unk_id
            feat_i = torch.full((features_lens[i], target_codec.shape[-1] - 1), self.speech_pad_id - 1)
            target_codec[i, : feat_i.shape[0], 1:] = feat_i
            target_codec[i, feat_i.shape[0], :] = eos_tensor
        target_codec = target_codec.to(torch.int)
        unpadded_target_texts = target_texts
        target_texts, target_text_lengths, target_texts_expanded = _convert_text_to_3d_tensor(target_texts)
        instructions, instruction_lengths, instructions_expanded_no_eos = _convert_text_to_3d_tensor(
            # tokens_to_generate is used in inference
            instructions,
            include_eos=False,
            tokens_to_generate=self.tokens_to_generate,
        )

        # answers = torch.concat([speaker_context, bos_tensor, target_codec], 1)

        # TODO: remove the following stanza
        if getattr(cuts[0], "s2s", False):
            # Add 1 for eos token
            token_list = [
                torch.concat([tt[: ttl + 1], tc[: tcl + 1]], 0)
                for tt, ttl, tc, tcl in zip(target_texts_expanded, target_text_lengths, target_codec, features_lens)
            ]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            # speech_loss_mask = torch.logical_and((tokens[:, :, 1:] != self.speech_unk_id), (tokens[:, :, 1:] != self.speech_pad_id))
            # text_loss_mask = torch.logical_and((tokens[:, :, 0:1] != text_unk_id), (tokens[:, :, 0:1] != text_pad_id))
            speech_loss_mask = tokens[:, :, 1:] != self.speech_pad_id
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = target_text_lengths + 1 + features_lens + 1 + instruction_lengths

        elif getattr(cuts[0], "s2s_align", False):
            bos_tensor = torch.full(
                (target_codec.shape[0], 1, self.n_speech_codebooks * self.decoder_reduction_factor + 1),
                self.speech_bos_id,
            ).to(torch.int)

            bos_tensor[:, :, 0] = self.text_processor.bos_id
            # [batch, max_feat_len]
            # the only thing needed is features_lens which can be estimated from target_audio length
            target_texts_expanded = self._expand_text_with_timestamps_and_word_lengths(
                unpadded_target_texts,
                word_lengths,
                start_time_tokens,
                features_lens + 1,
                self.codec_model_downsampling_factor / self.codec_sample_rate,
                pad_id=text_unk_id,
            )
            # [batch, max_feat_len, 1+V], where V = #codebooks * reduction_factor
            if target_texts_expanded.shape[0] == target_codec.shape[0]:
                target_codec[:, :, 0] = target_texts_expanded
            else:
                raise ValueError("target_texts_expanded and target_codec have different batch size")
            token_list = torch.concat([bos_tensor, target_codec], 1)
            features_lens += 1

            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)
            speech_loss_mask = tokens[:, :, 1:] != self.speech_pad_id
            # Make the text loss mask the same as speech since they are aligned
            loss_mask = torch.cat([speech_loss_mask[..., :1], speech_loss_mask], dim=-1)
            if not self.t5_style:
                for itl in instruction_lengths:
                    loss_mask[:, :itl, :] = False
            # loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            # full_lengths = target_text_lengths + 1 + features_lens + 1 + instruction_length
            full_lengths = features_lens + 1 + instruction_lengths
            target_text_lengths = -1 * torch.ones_like(target_text_lengths)  # bos_tensor
        elif getattr(cuts[0], "direct_s2s", False):
            # Add 1 for eos token
            # tt[0] is the bos token
            token_list = [
                torch.concat([tt[:1], tc[: tcl + 1]], 0)
                for tt, tc, tcl in zip(target_texts_expanded, target_codec, features_lens)
            ]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            speech_loss_mask = tokens[:, :, 1:] != self.speech_pad_id
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = 1 + features_lens + 1 + instruction_lengths
        elif getattr(cuts[0], "s2t", False):
            # Add 1 for eos token
            token_list = [tt[: ttl + 1] for tt, ttl in zip(target_texts_expanded, target_text_lengths)]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            speech_loss_mask = torch.zeros(tokens.shape[0], tokens.shape[1] - 1, tokens.shape[2])
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = target_text_lengths + 1 + instruction_lengths
        full_lengths = torch.clamp(full_lengths, max=tokens.shape[1])
        # simplify above code
        # Start from index 1 since the first token will not be used as a label
        loss_mask = loss_mask[:, 1:, :]

        # Merge batch
        # note: the codec id in labels and contexts and others do not consider the offset e.g. speech_eos is 1002
        # the offset is all considered by SumVocabParallelEmbedding
        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,
            "audio_signal_length": audio_lens,
            "metadata": metadata,
            # For forward
            "instructions": instructions,
            "target_texts_expanded": target_texts_expanded,  # used in prepare_llm_input
            "contexts": instructions_expanded_no_eos,  # used in inference
            "context_lengths": instruction_lengths,
            "tokens": tokens[:, :-1, :],
            "tokens_length": full_lengths - 1,
            "labels": tokens[:, 1:, :],
            "loss_mask": loss_mask,
            # For validation mainly
            "target_texts": target_texts,
            "target_text_lengths": target_text_lengths,
            "answers": tokens[:, 1:, :],
            "answer_audio": answer_audios,
            "answer_audio_lens": answer_audio_lens,
            "num_turns": torch.Tensor(num_turns).long(),
            "answer_features_lens": torch.Tensor(features_lens).long(),
        }

        return return_batch
