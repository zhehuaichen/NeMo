# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.models.duplex_s2s_speech_decoder_model import DuplexS2SSpeechDecoderModel # Import parent class
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class DuplexAsr2SModel(DuplexS2SSpeechDecoderModel):

    def prepare_inputs(self, batch: dict):
        """
        Specific input preparation for DuplexAsr2SModel's streaming ASR mode.
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """

        inputs = super().prepare_inputs(batch)  # Call parent to set up basic inputs
        inputs_t2t = self.prepare_t2t_inputs(batch)  # Prepare T2T specific inputs
        inputs.update(inputs_t2t)
        return inputs

    def prepare_t2t_inputs(self, batch: dict):
        """
        Prepare inputs for text-to-text processing.
        """
        source_tokens = batch["source_tokens"]
        target_tokens = batch["target_tokens"]
        
        # For T2T, typically input is source, label is target.
        # Autoregressive: input is target_shifted_right, label is target.
        # Here, it seems to combine source and target for input.
        
        # Assuming standard autoregressive T2T: input is target[:, :-1], labels are target[:, 1:]
        # And source_tokens are used to condition.
        
        text_inputs = target_tokens[:, :-1]
        
        agent_embeds = self.embed_tokens(text_inputs)
        
        # Align source_tokens with text_inputs length for combination
        source_tokens_aligned = source_tokens[:, :text_inputs.shape[1]]
        user_embeds = self.embed_tokens(source_tokens_aligned)
        
        input_embeds = self._combine_embeddings(agent_embeds, user_embeds)
        
        return {
            "input_embeds": input_embeds,
        }

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Get BOS embedding. For T2T, this is a text BOS.
        For S2S/ASR (self.offline_inference), it's used for the text channel.
        The original DuplexAsr2SModel used text_pad_id.
        Using text_bos_id seems more standard for a "beginning of sequence".
        """
        bos_token_id = self.text_bos_id if self.text_bos_id is not None else self.text_pad_id
        text_bos = torch.full((1,), fill_value=bos_token_id, device=self.device, dtype=torch.long)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    def _combine_embeddings(self, agent_embeds: Tensor, user_embeds: Tensor) -> Tensor:
        """
        Combine agent and user embeddings based on the configuration.
        
        Args:
            agent_embeds: Agent (target) embeddings of shape (B, T, H)
            user_embeds: User (source) embeddings of shape (B, T, H)
            
        Returns:
            Combined embeddings of shape (B, T, H)
        """
        if False:
            # Concatenate embeddings and project to original dimension
            concatenated = torch.cat([agent_embeds, user_embeds], dim=-1)  # (B, T, 2*H)
            return self.concat_projection(concatenated)  # (B, T, H)
        else:
            # Add embeddings (original behavior)
            user_weighted = user_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
            return agent_embeds + user_weighted

    def validation_step(self, batch: dict, batch_idx: int):

        # Update speaker embedding to reflect the one in the prompt during inference
        if self.speech_generation.use_speaker_encoder and self.speech_generation.inference_speaker_reference:
            self.speech_generation.update_inference_speaker_embedding(self.speech_generation.inference_speaker_reference)

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_t2t_inference(
                dataset_batch["source_tokens"],
                dataset_batch["source_tokens_len"],
            )
            #TODO: add speech generation and then call asr_bleu

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                '''asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )'''

                # TODO: add speech generation result to the following logger
                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=dataset_batch["target_texts"],  # TODO
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=dataset_batch["target_audio"],  # TODO
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

    @torch.no_grad()
    def offline_t2t_inference(
        self,
        source_tokens: torch.Tensor,
        source_lens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive text-to-text prediction.
        """
        B, T_src = source_tokens.shape
        
        source_embeds = self.embed_tokens(source_tokens)
        
        # Max generation length, e.g., same as source or a config param.
        # For simplicity, T_gen = T_src. Add EOS handling for variable length.
        T_gen = T_src 
        
        cache = DynamicCache()
        gen_tokens = torch.empty(B, T_gen, device=self.device, dtype=torch.long)
        
        # First step: agent is BOS, user is first source token embedding
        current_agent_embedding = self._get_bos_embedding().expand(B, 1, -1)

        for t in range(T_gen):
            current_source_embedding_idx = min(t, T_src - 1)
            current_source_embed = source_embeds[:, current_source_embedding_idx:current_source_embedding_idx+1]
            
            step_embeds = self._combine_embeddings(current_agent_embedding, current_source_embed)
            
            # Pass cache from previous step if t > 0
            current_cache = ans.get("cache") if t > 0 else cache 
            ans = self(step_embeds, cache=current_cache) 
            
            predicted_token = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_tokens[:, t] = predicted_token
            
            current_agent_embedding = self.embed_tokens(predicted_token.unsqueeze(1))

            # TODO: EOS handling to stop generation and adjust generated_lens
        
        generated_lens = source_lens.clone() # Placeholder

        return {
            "text": tokens_to_str(gen_tokens, generated_lens, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens": gen_tokens,
            "tokens_len": generated_lens,
        }
