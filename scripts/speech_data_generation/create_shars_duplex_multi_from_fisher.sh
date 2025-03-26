#!/bin/bash
#SBATCH -A llmservice_nemo_mlops
#SBATCH -J "llmservice_nemo_mlops:enhanced_fisher_create_shars"
#SBATCH -N 1 # number of nodes
#SBATCH -t 04:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=1    # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# autorun.sh -n 1 ./create_shars_duplex_multi_from_fisher.sh -m callfriend_eng_south2_LDC2020S08 -s 10
# for i in `seq 1 1 511`; do  autorun -n 1 "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex3/scripts/speech_data_generation/create_shars_duplex_multi_from_shar.sh $i"; done
set -x
SLURM_ACCOUNT=portfolios/llmservice
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/${SLURM_ACCOUNT}  
USERID="users/vtrinh"
CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/projects/speechllm_full_duplex_with_voice_prompt/code/NeMo_s2s_duplex3
NUM_SHARD=10

MOUNTS="--container-mounts=/lustre/:/lustre/,$CODE_DIR/:/code,/lustre/fsw/portfolios/llmservice/${USERID}/results/:/results,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data:/data,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12,/lustre/fsw/portfolios/llmservice/${USERID}/results/HFCACHE/:/hfcache/"
PROJECT_NAME=create_shars
EXP_NAME=create_shars_enhanced_fisher

RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME

mkdir -p ${RESULTS_DIR}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# Parse command line arguments
while getopts "m:a:o:s:" opt; do
  case $opt in
    m) MANIFEST="$OPTARG"
    ;;
    a) AUDIO_PATH="$OPTARG"
    ;;
    o) OUT_SHAR_DIR="$OPTARG"
    ;;
    s) NUM_SHARD="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        echo "Usage: $0 -m manifest_path -a audio_path -o output_shar_dir -s num_shards" >&2
        exit 1
    ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MANIFEST" ] || [ -z "$AUDIO_PATH" ] || [ -z "$OUT_SHAR_DIR" ] || [ -z "$NUM_SHARD" ]; then
    echo "Usage: $0 -m manifest_path -a audio_path -o output_shar_dir -s num_shards"
    echo "Example: $0 -m /path/to/manifest.ndjson -a /path/to/audio -o /path/to/output -s 10"
    exit 1
fiß

# Execute the command directly with provided arguments
cmd="
python ${CODE_DIR}/scripts/speech_data_generation/create_shars_duplex_multi_from_fisher_parallel.py \
    --manifest ${MANIFEST} \
    --audio_path ${AUDIO_PATH} \
    --out_shar_dir ${OUT_SHAR_DIR} \
    --num_shard ${NUM_SHARD}
"
srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"








































































































































































































