eval代码执行脚本:
```bash
python3 scripts/custom_batch_eval.py \
  --output_dir "./eval" \
  --eval_file "/home/wangt/corag_dev/Agentic_CoRAG-corag/data/rejection_sampled_data.json" \
  --save_file "eval/rejection_sampled_data_eval_out.json" \
  --vllm_api_base "http://10.44.124.218:8000/v1" \
  --final_answer_api_base "http://10.44.124.218:8000/v1" \
  --sub_answer_api_base "http://10.44.124.218:8000/v1" \
  --vllm_api_key "token-123" \
  --final_answer_api_key "token-123" \
  --sub_answer_api_key "token-123" \
  --vllm_model "qwen-2.5-7b-instruct" \
  --tokenizer_name "/home/data/Qwen2.5-7B-Instruct" \
  --decode_strategy "greedy" \
  --max_path_length 3 \
  --num_threads 2 \
  --calc_recall true \
  --enable_naive_retrieval true
```