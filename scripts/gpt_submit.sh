if [ $# -lt 2 ]; then
    echo "Error: Please provide an identifier as an argument"
    echo "Usage: sbatch $0 <identifier> <task-type>" 
    exit 1
fi

IDENTIFIER="$1"
TASK_TYPE="$2"

INPUT_DIR="./outputs/${IDENTIFIER}/"
OUTPUT_DIR="./outputs/${IDENTIFIER}/"

cd ~/TinyZero

python behavioral_evals/gpt_api_eval.py \
    --input "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --api-key api_key \
    --num-samples 200 \
    --task-type "${TASK_TYPE}"