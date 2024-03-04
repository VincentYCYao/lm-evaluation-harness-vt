# Command line variables
model_name=$1
task=$2
batch_size=$3
save_file_name=$4
include_path=$5


python lm_eval \
        --model hf \
        --model_args pretrained=${model_name},parallelize=True \
        --tasks ${task} \
        --device cuda \
        --batch_size ${batch_size} \
        --output_path /app/lm-eval-log/${save_file_name} \
        --include_path ${include_path} \
        --log_samples
