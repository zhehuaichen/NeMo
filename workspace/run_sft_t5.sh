NEMO_DIR=/workspace/nemo/works/mod_speech_llm/NeMo
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
MEGATRON_CKPT=/lustre/fsw/swdl/swdl-langspeech/sandeepsub/models/t5_220m/megatron_t5--val_loss-1.47-step-999999-consumed_samples-2047944704.0.nemo
ASR_MODEL="ssl_en_conformer_large"
ASR_MODEL="stt_en_fastconformer_transducer_large"
GLOBAL_BATCH=2
MICRO_BATCH=2

TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_140_r.shuf.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_11.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10_text.json,/media/data/datasets/LibriSpeech/dev_clean_10_text.json]
train_questions=[/media/data/datasets/LibriSpeech/dev_clean_10_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10_text.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_10_text.json]
train_questions=[/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_2.json.diff]
train_questions=[/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_2.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json

VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_11.json]
valid_questions=[/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_10_text.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_2.json.diff]
valid_questions=[/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]

 export HF_HOME="/hfcache/" 
 export HF_DATASETS_CACHE="/hfcache/datasets" 
 export TRANSFORMERS_CACHE="/hfcache/models" 
python -m pdb -c continue \
run_sft_audio_lm_t5.py --config-path="../examples/multimodel/conf/speechllm/" --config-name "modularized_speech_t5_config" \
    model.pretrained_audio_model=$ASR_MODEL \
    model.language_model_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++model.vocab_file=/tmp/megatron-bert-345m-cased_vocab  \
    ++model.data.train_ds.question_file_set=$train_questions \
    ++model.data.train_ds.random_context_prob=0.5 \
    ++model.data.train_ds.random_context_num=64 \
    ++model.data.validation_ds.question_file_set=$valid_questions \
    ++model.data.validation_ds.random_context_prob=0.5 \
    ++model.data.validation_ds.random_context_num=64 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS

