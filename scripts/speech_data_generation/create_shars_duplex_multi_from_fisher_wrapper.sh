#!/bin/bash

# Base paths
MANIFEST_BASE="/lustre/fsw/portfolios/llmservice/users/vtrinh/cs-oci-mirror/lustre/fs12/portfolios/adlr/projects/adlr_audio_speech/datasets/duplex_speech/transcripts/manifests"
AUDIO_BASE="/lustre/fsw/portfolios/adlr/users/rajarshir/duplexspeech/dialog_audio_datasets/enhanced/stages/ln_dm_vf_dm_mb_df"
OUT_SHAR_BASE="/lustre/fsw/portfolios/llmservice/users/vtrinh/datasets/s2s_shar/duplex/enhanced_fisher"

# List of manifest names
MANIFEST_NAMES=(
    #"callfriend_eng_south2_LDC2020S08"
    #"callhome_eng_LDC97S42"
    "fisher_eng_p1_LDC2004S13"
    "fisher_eng_p2_LDC2005S13"
)

# List of number of shards for each dataset
NUM_SHARDS_LIST=(
    #10  # for callfriend_eng_south2_LDC2020S08
    #10  # for callhome_eng_LDC97S42
    6  # for fisher_eng_p1_LDC2004S13
    6  # for fisher_eng_p2_LDC2005S13
)

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_SCRIPT="${SCRIPT_DIR}/create_shars_duplex_multi_from_fisher.sh"

# Process each dataset
for idx in "${!MANIFEST_NAMES[@]}"; do
    MANIFEST_NAME="${MANIFEST_NAMES[$idx]}"
    NUM_SHARDS="${NUM_SHARDS_LIST[$idx]}"
    AUDIO_PATH="${AUDIO_BASE}/${MANIFEST_NAME}"
    
    # Handle fisher datasets differently
    if [[ $MANIFEST_NAME == fisher_eng_p* ]]; then
        # Process each manifest shard for fisher datasets
        for i in {0..9}; do
            # Format number with leading zeros (6 digits)
            MANIFEST_NUM=$(printf "%06d" $i)
            MANIFEST_FILE="${MANIFEST_NAME}/manifest_${MANIFEST_NUM}.ndjson"
            MANIFEST="${MANIFEST_BASE}/${MANIFEST_FILE}"
            OUT_SHAR_DIR="${OUT_SHAR_BASE}/${MANIFEST_NAME}/manifest_${MANIFEST_NUM}"
            
            echo "Processing Fisher dataset:"
            echo "  Manifest: $MANIFEST"
            echo "  Audio Path: $AUDIO_PATH"
            echo "  Output Path: $OUT_SHAR_DIR"
            echo "  Number of Shards: $NUM_SHARDS"
            
            # Submit the job using sbatch
            sbatch "$MAIN_SCRIPT" \
                -m "$MANIFEST" \
                -a "$AUDIO_PATH" \
                -o "$OUT_SHAR_DIR" \
                -s "$NUM_SHARDS"
            
            # Add a small delay between submissions
            sleep 1
        done
    else
        # Process regular datasets
        MANIFEST_FILE="${MANIFEST_NAME}.ndjson"
        MANIFEST="${MANIFEST_BASE}/${MANIFEST_FILE}"
        OUT_SHAR_DIR="${OUT_SHAR_BASE}/${MANIFEST_NAME}"
        
        echo "Processing regular dataset:"
        echo "  Manifest: $MANIFEST"
        echo "  Audio Path: $AUDIO_PATH"
        echo "  Output Path: $OUT_SHAR_DIR"
        echo "  Number of Shards: $NUM_SHARDS"
        
        # Submit the job using sbatch
        sbatch "$MAIN_SCRIPT" \
            -m "$MANIFEST" \
            -a "$AUDIO_PATH" \
            -o "$OUT_SHAR_DIR" \
            -s "$NUM_SHARDS"
        
        # Add a small delay between submissions
        sleep 1
    fi
done