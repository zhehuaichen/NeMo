set -x

NEMO_DIR=/workspace/nemo/works/zhehuaic_works/tts/NeMo/
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
HYDRA_FULL_ERROR=1 

name=T5_1_18_24.1a
mkdir -p tts_results/$name

codec=/home17/jasoli/models/SpeechCodec.nemo
codec=/home17/lab/models/T5_1_18_24/SpeechCodec.nemo
cd $NEMO_DIR 
export CUDA_VISIBLE_DEVICES=1
python $NEMO_DIR/examples/nlp/language_modeling/megatron_t5_speechlm_sft_inference.py \
--config-name=megatron_t5_speechlm_inference.yaml \
name=$name \
model.data.test_ds='["/home17/jasoli/data_prime/RIVA-TTS/Rivatts_AllLanguages_01_30_24_val_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts_doci_frenchphones.json"]' \
exp_manager.exp_dir=examples/asr/NeMo_experiments/tts_results/$name \
+model.codecmodel_path=$codec \
checkpoint_path=/home17/lab/models/T5_1_18_24/CTC_All_Data_Phoneme_lr5e-5_Parallel_step218269.ckpt \
model.language_model_path=/home17/lab/models/T5_1_18_24/megatron_t5_expanded_vocab_posemb1536.nemo \
model.data.sup_data_path=/home17/jasoli/data_prime/RIVA-TTS/ \
+model.data.codec_folder="/home17/jasoli/data_prime/RIVA-TTS/codecs/" \
model.data.train_task=all \
+model.freeze_model=False \
model.data.max_seq_length=1536 \
model.max_inference_timesteps=1500 \
+model.data.context_duration_min=2.9 \
+model.data.context_duration_max=2.9 \
+model.data.context_pattern=parallel \
model.top_k=80 \
model.temperature=0.8 \
model.global_batch_size=2 \
model.micro_batch_size=2 \
model.data.speech_offset=30128 \
+model.data.num_speech_codebooks=8 \
+model.data.codebook_fps=86 \
+model.codecmodel_type=nemo_codec \
+model.data.lm_vocab_size=30000 \
trainer.devices=1 \
trainer.precision=16 \
model.seq_pattern=parallel \
+model.english_only_model=True \
~model.data.g2p \
model.speech_head_type=linear
cd -

# +model.num_sentinel_tokens=9775 +model.override_token_model=/home/jasoli/models/megatron_mt5_expanded_vocab_posemb/mt5_tokenizer_w_phones_v3_01_26_24.model model.data.speech_offset=250322 +model.lm_vocab_size=250322 +model.data.context_duration_max=2.9 +model.data.context_duration_min=2.9 +model.data.add_special_tokens_to_only_first_codebook=True trainer.devices=1   'model.data.grapheme_prefix="&"' 
