set -x

NEMO_DIR=/workspace/nemo/works/zhehuaic_works/tts/NeMo/
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

name=T5_1_18_24.1b
mkdir -p examples/multimodal/NeMo_experiments/tts_results/$name
BATCH=2

codec=/home17/jasoli/models/SpeechCodec.nemo
codec=/home17/lab/models/T5_1_18_24/SpeechCodec.nemo
cd $NEMO_DIR 

VAL_MANIFESTS="
 	manifests/dev_mcv12_manifest_pcstrip_es_2k.40.json
	manifests/mcv11_dev_clean_pcstrip_en_2k.40.json \


"

VAL_MANIFESTS="
	manifests/covost_v2.es_en.dev.es.40.json \

	 "
	model=/workspace/nemo/works/zhehuaic_works/llm/oci_b6s4kf-ASR-AST_20240104_lfbe-128_ngpus-128_mbs-240s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-100000-averaged.nemo
	model=/workspace/nemo/works/zhehuaic_works/llm/oci_audiosetnoise_langtemp0.2_dsettemp0.5_20240124_kd_lfbe-128_ngpus-128_mbs-360s_sunoInit__opt-adamw_lr-2e-5_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-75000--CLEANED.nemo
	model=/workspace/nemo/works/zhehuaic_works/llm/oci_b6s4kf-sunolong_noCC_langtemp0.5_dsettemp0.5_20240126_lfbe-128_ngpus-128_mbs-360s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-150000--CLEANED.nemo
	odir=examples/multimodal/NeMo_experiments/tts_results/$name
	mkdir -p $odir
	HYDRA_FULL_ERROR=1
	for VAL in $VAL_MANIFESTS; do
		output_filename=$odir/`basename $VAL`
	#gdb -ex r --args python \

python -Xfaulthandler  -m pdb -c continue \
  $NEMO_DIR/examples/multimodal/modular_speechllm/s2s_inference.py \
--config-name=s2s_t5_inference.yaml \
--config-path=$NEMO_DIR/examples/multimodal/modular_speechllm/conf \
  stt_model.model_path=$model  stt_model.dataset_manifest=examples/multimodal/NeMo_experiments/$VAL \
  ++stt_model.multitask_decoding.beam.return_best_hypothesis=false \
  ++stt_model.multitask_decoding.beam.beam_size=5 \
  ++stt_model.use_model_transcribe=True \
  ++stt_model.gt_text_attr_name="answer" \
  ++stt_model.output_filename=$output_filename \
  ++stt_model.batch_size=$BATCH \
name=$name \
model.data.test_ds='["/home17/jasoli/data_prime/RIVA-TTS/Rivatts_AllLanguages_01_30_24_val_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts_doci_frenchphones.json"]' \
exp_manager.exp_dir=$odir \
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
+model.data.context_duration_max=6.9 \
+model.data.context_pattern=parallel \
model.top_k=80 \
model.temperature=0.8 \
model.global_batch_size=$BATCH \
model.micro_batch_size=$BATCH \
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

done
