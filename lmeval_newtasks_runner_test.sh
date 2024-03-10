python  lm_eval \
        --model hf \
        --model_args pretrained=meta-llama/Llama-2-13b-chat-hf,parallelize=True \
        --tasks biolama_ctd \
        --device mps \
        --batch_size auto \
        --output_path /Users/vincent/Documents/GP-NLP/lm-eval-log/llama-2-13b-chat-biolama-ctd \
        --include_path /Users/vincent/Documents/GP-NLP/lm-evaluation-harness-vt/probing_tasks_vt \
        --log_samples\
        --limit 50
