#cur="/lustre/fsw/swdl/swdl-langspeech/datasets/data/speechlm_sft/msmarco_train_not_normalized.json"
cur=input/qa_datasets/squadv2_validation_normalized.json
cur=input/qa_datasets/speech_instruction.json
cur=input/qa_datasets/msmarco_test_normalized.json
cur=input/qa_datasets/msmarco_train_normalized.json
cur=/workspace/nemo/works/zhehuaic_works/data/alpaca.jsonl
#-m pdb -c continue 
#time python sft_tts_generate_spec.py $cur 2>&1 | tee outputs/`basename $cur`.log
export CUDA_VISIBLE_DEVICES=1
time python sft_tts_generate_spec_speechall.py $cur 2>&1 | tee outputs_speechall/`basename $cur`.log

# scp -r draco-rno-login:/gpfs/fs1/projects/ent_aiapps/users/vnoroozi/data/llm_sft_asr/qa_datasets/ input/
