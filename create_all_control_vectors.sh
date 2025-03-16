#!/bin/bash

# Usage examples:
# ./create_all_control_vectors.sh "0" "./aya-23-35B" "aya-23:35b-" 8192 8 true 4bit
# ./create_all_control_vectors.sh "1" "./Qwen1.5-14B-Chat" "qwen-1.5:14b-" 5120 4 true 8bit
# ./create_all_control_vectors.sh "0,1" "./c4ai-command-r-plus" "command-r-plus:104b-" 12288 2 true none

# Check if we have the correct number of arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <cuda_devices> <model_id> <output_prefix> <num_prompt_samples> [batch_size=1] [use_bfloat16=true] [quantization=4bit]"
    echo "Example: $0 \"0,1\" \"/path/to/model\" \"model-prefix-\" 12345 8 true"
    exit 1
fi

# Assuming the 'data' sub-folder is in the default location.
DATA="data"
STEMS="$DATA/prompt_stems.json"
PROMPTS="$DATA/writing_prompts.txt"

# Assign arguments to variables
CUDA_DEVICES="$1"
MODEL_ID="$2"
OUTPUT_PREFIX="$3"
NUM_PROMPT_SAMPLES="$4"
BATCH_SIZE="${5:-1}"
USE_BFLOAT16="${6:-true}"
QUANTIZATION="${7:-4bit}"

if [ "$USE_BFLOAT16" = "true" ]; then
    BFLOAT16_FLAG="--use_bfloat16"
else
    BFLOAT16_FLAG=""
fi

# Define arrays for continuations and output suffixes
continuations=(
    "$DATA/writing_style_continuations/character_focus.json"
    "$DATA/writing_style_continuations/language.json"
    "$DATA/writing_style_continuations/storytelling.json"
    "$DATA/dark_tetrad_continuations/compassion_vs_sadism.json"
    "$DATA/dark_tetrad_continuations/empathy_vs_sociopathy.json"
    "$DATA/dark_tetrad_continuations/honesty_vs_machiavellianism.json"
    "$DATA/dark_tetrad_continuations/humility_vs_narcissism.json"
    "$DATA/other_continuations/optimism_vs_nihilism.json"
)

output_suffixes=(
    "character_focus_"
    "language_"
    "storytelling_"
    "compassion_vs_sadism_"
    "empathy_vs_sociopathy_"
    "honesty_vs_machiavellianism_"
    "humility_vs_narcissism_"
    "optimism_vs_nihilism_"
)

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Loop through continuations and create control vectors
for i in "${!continuations[@]}"; do
    python3 ./create_control_vectors.py \
        --model_id "$MODEL_ID" \
        --output_path "${OUTPUT_PREFIX}${output_suffixes[i]}" \
        --prompt_stems_file "$STEMS" \
        --writing_prompts_file "$PROMPTS" \
        --continuations_file "${continuations[i]}" \
        --num_prompt_samples "$NUM_PROMPT_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --quantization "$QUANTIZATION" \
        $BFLOAT16_FLAG
done