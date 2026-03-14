
2026/3/13  17:25

## 基于原生qwen2.5-7b-instruct模型的性能测试

```bash
=== Eval Analysis ===
Samples: 152

[Answer Quality]
EM: 0.1776
Token F1: 0.2473
Contains Match: 0.2237
Yes/No Accuracy: 0.2778 (n=36)
No Relevant Information Rate: 0.0855

[Timing]
path_generation: mean=26.79s, median=23.63s, p90=27.41s, p95=41.55s, max=151.79s
final_generation: mean=0.33s, median=0.30s, p90=0.48s, p95=0.55s, max=0.70s
end_to_end: mean=27.12s, median=23.98s, p90=27.64s, p95=41.86s, max=152.13s

[Reasoning]
avg_steps=2.50, median_steps=3.00, min_steps=1, max_steps=3

[Retrieval Diagnostics]
zero_corag_recall_rate=0.2697
no_info_rate=0.0855
yes_no_error_rate=0.7222
relation_drift_count=2

[CoRAG Recall]
micro=0.5138, macro=0.5122, hits=168, gold=327

[Naive Recall]
micro=0.4098, macro=0.4056, hits=134, gold=327
Recall comparison: corag_better=42, naive_better=10, tie=100

[Slowest Samples]
1. total=152.13s, path=151.79s, final=0.34s, steps=3, question=Were both Pietro Salini and Domenico Ravenna, born in the same place?
2. total=142.30s, path=141.92s, final=0.38s, steps=3, question=Where did the composer of film The Straw Hat die?
3. total=120.03s, path=119.53s, final=0.51s, steps=3, question=Who is the father-in-law of Sisowath Kossamak?
4. total=92.13s, path=91.73s, final=0.40s, steps=2, question=Who is the child of the performer of song Me And Bobby Mcgee?
5. total=90.90s, path=90.60s, final=0.29s, steps=2, question=Are both movies, Naked Tango and Algiers (Film), from the same country?
6. total=66.28s, path=65.67s, final=0.62s, steps=3, question=Who is the child of the director of film An Event?
7. total=60.32s, path=59.97s, final=0.35s, steps=2, question=Where did Saw Thanda's husband die?
8. total=43.21s, path=42.85s, final=0.37s, steps=3, question=Where was the director of film The Outlaw Express born?
9. total=40.76s, path=40.50s, final=0.26s, steps=3, question=Are Christopher Newton (Criminal) and Frances M. Vega of the same nationality?
10. total=37.07s, path=36.84s, final=0.24s, steps=2, question=Did the movies Inside The Room and Crude Set Drama, originate from the same country?

[Notes]
- The original eval summary stores path generation time in the third slot of time[].
- In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.
- avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.
- Low recall often comes from subquery relation drift or overly broad entity resolution before retrieval, not only from the retriever itself.
```



运行日志
```bash
Input eval file: /home/wangt/corag_dev_lmx/Agentic_CoRAG-corag/eval/2wikimultihopqa_hard_small_eval_output.json
=== Eval Analysis ===
Samples: 24

[Answer Quality]
EM: 0.0000
Token F1: 0.0343
Contains Match: 0.3333
Yes/No Accuracy: 0.0000 (n=8)
No Relevant Information Rate: 0.5417

[Timing]
path_generation: mean=29.07s, median=28.68s, p90=32.69s, p95=34.85s, max=36.10s
final_generation: mean=1.90s, median=1.79s, p90=3.70s, p95=3.75s, max=3.78s
end_to_end: mean=30.97s, median=30.93s, p90=35.99s, p95=38.17s, max=38.84s

[Reasoning]
avg_steps=2.88, median_steps=3.00, min_steps=2, max_steps=3

[Retrieval Diagnostics]
zero_corag_recall_rate=0.5833
no_info_rate=0.5417
yes_no_error_rate=1.0000
relation_drift_count=1

[CoRAG Recall]
micro=0.1803, macro=0.2062, hits=11, gold=61

[Naive Recall]
micro=0.0164, macro=0.0104, hits=1, gold=61
Recall comparison: corag_better=10, naive_better=1, tie=13

[Slowest Samples]
1. total=38.84s, path=35.08s, final=3.75s, steps=3, question=Who is younger, Denise Kandel or Bruce Robinson?
2. total=38.44s, path=36.10s, final=2.34s, steps=3, question=Who is the child of the performer of song Me And Bobby Mcgee?
3. total=36.65s, path=33.50s, final=3.15s, steps=3, question=What is the date of birth of Henry I Of Ziębice's father?
4. total=34.45s, path=30.80s, final=3.65s, steps=3, question=Which award the director of film Edelweiss Pirates (Film) received?
5. total=33.39s, path=29.66s, final=3.73s, steps=3, question=Who is the maternal grandmother of Prince Dmitri Alexandrovich Of Russia?
6. total=32.54s, path=30.73s, final=1.80s, steps=3, question=Do both films Once Upon A Time In America and Naked Violence (Film) have the directors that share the same nationality?
7. total=32.18s, path=30.00s, final=2.18s, steps=3, question=Who was born earlier, Elmer W. Conti or Seth Joshua?
8. total=31.96s, path=29.20s, final=2.76s, steps=3, question=Who is Princess Mafalda Of Savoy's maternal grandmother?
9. total=31.77s, path=29.99s, final=1.78s, steps=3, question=Who is Anne Gust Brown's father-in-law?
10. total=31.63s, path=30.26s, final=1.37s, steps=3, question=Who is Marie Zéphyrine Of France's paternal grandmother?

[Notes]
- The original eval summary stores path generation time in the third slot of time[].
- In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.
- avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.
- Low recall often comes from subquery relation drift or overly broad entity resolution before retrieval, not only from the retriever itself.
```


---

## 基于微调后的qwen2.5-7b-instruct模型的性能测试

完整数据集性能参数结果:

```bash
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# python3 scripts/analyze_custom_eval.py
Input eval file: /home/wangt/corag_dev_lmx/Agentic_CoRAG-corag/eval/2wikimultihopqa_hard_full_eval_output_3_13_17_43.json
=== Eval Analysis ===
Samples: 152

[Answer Quality]
EM: 0.0000
Token F1: 0.0411
Contains Match: 0.2105
Yes/No Accuracy: 0.0000 (n=36)
No Relevant Information Rate: 0.5921

[Timing]
path_generation: mean=28.47s, median=23.93s, p90=27.58s, p95=29.63s, max=383.93s
final_generation: mean=1.94s, median=1.97s, p90=3.36s, p95=3.65s, max=4.02s
end_to_end: mean=30.41s, median=25.93s, p90=29.69s, p95=31.37s, max=386.76s

[Reasoning]
avg_steps=2.62, median_steps=3.00, min_steps=1, max_steps=3

[Retrieval Diagnostics]
zero_corag_recall_rate=0.5987
no_info_rate=0.5921
yes_no_error_rate=1.0000
relation_drift_count=4

[CoRAG Recall]
micro=0.2599, macro=0.2694, hits=85, gold=327

[Naive Recall]
micro=0.0550, macro=0.0576, hits=18, gold=327
Recall comparison: corag_better=53, naive_better=5, tie=94

[Slowest Samples]
1. total=386.76s, path=383.93s, final=2.83s, steps=3, question=Who is William Ii, Prince Of Nassau-Dillenburg's paternal grandfather?
2. total=382.73s, path=380.01s, final=2.72s, steps=2, question=Who is the paternal grandfather of Bernhard Iv, Prince Of Anhalt-Bernburg?
3. total=37.47s, path=33.45s, final=4.02s, steps=3, question=Who is the paternal grandfather of Richard Beauchamp, 1St Earl Of Worcester?
4. total=33.15s, path=29.84s, final=3.31s, steps=2, question=Who is Charles Willoughby, 10Th Baron Willoughby Of Parham's paternal grandfather?
5. total=33.01s, path=31.52s, final=1.49s, steps=3, question=Who is James Lyon, 7Th Earl Of Strathmore And Kinghorne's paternal grandfather?
6. total=32.17s, path=28.49s, final=3.68s, steps=3, question=Who is younger, Denise Kandel or Bruce Robinson?
7. total=31.81s, path=31.20s, final=0.62s, steps=3, question=Where did Adolph Of Cleves, Lord Of Ravenstein's mother die?
8. total=31.54s, path=31.02s, final=0.51s, steps=3, question=Were Jimmy Santiago Baca and Duane Armstrong from the same country?
9. total=31.24s, path=30.71s, final=0.52s, steps=3, question=Are Antoine Jean-Baptiste Thomas and Canardo (Rapper) of the same nationality?
10. total=30.81s, path=26.99s, final=3.82s, steps=3, question=Who is the maternal grandmother of Prince Dmitri Alexandrovich Of Russia?

[Notes]
- The original eval summary stores path generation time in the third slot of time[].
- In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.
- avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.
- Low recall often comes from subquery relation drift or overly broad entity resolution before retrieval, not only from the retriever itself.
```


---

3/13 22:15 small数据集性能测试:

说明：这一段当前文档中的结果块误引用了完整集文件 `2wikimultihopqa_hard_full_eval_output_3_13_17_43.json`，并且 `Samples` 也错误地显示为 `152`。  
本轮代码已经补齐 `custom_batch_eval.py` 与 `analyze_custom_eval.py` 的诊断字段和公平性口径；small 集结果需要基于更新后的代码在服务器上重新运行后，再用新的真实输出覆盖本段。

```bash
Input eval file: /home/wangt/corag_dev_lmx/Agentic_CoRAG-corag/eval/2wikimultihopqa_hard_full_eval_output_3_13_17_43.json
=== Eval Analysis ===
Samples: 152

[Answer Quality]
EM: 0.0592
Token F1: 0.1437
Contains Match: 0.1513
Yes/No Accuracy: 0.1944 (n=36)
No Relevant Information Rate: 0.0921

[Timing]
path_generation: mean=28.47s, median=23.93s, p90=27.58s, p95=29.63s, max=383.93s
final_generation: mean=1.94s, median=1.97s, p90=3.36s, p95=3.65s, max=4.02s
end_to_end: mean=30.41s, median=25.93s, p90=29.69s, p95=31.37s, max=386.76s

[Reasoning]
avg_steps=2.62, median_steps=3.00, min_steps=1, max_steps=3

[Retrieval Diagnostics]
zero_corag_recall_rate=0.5987
no_info_rate=0.0921
yes_no_error_rate=0.8056
relation_drift_count=4

[CoRAG Recall]
micro=0.2599, macro=0.2694, hits=85, gold=327

[Naive Recall]
micro=0.0550, macro=0.0576, hits=18, gold=327
Recall comparison: corag_better=53, naive_better=5, tie=94

[Slowest Samples]
1. total=386.76s, path=383.93s, final=2.83s, steps=3, question=Who is William Ii, Prince Of Nassau-Dillenburg's paternal grandfather?
2. total=382.73s, path=380.01s, final=2.72s, steps=2, question=Who is the paternal grandfather of Bernhard Iv, Prince Of Anhalt-Bernburg?
3. total=37.47s, path=33.45s, final=4.02s, steps=3, question=Who is the paternal grandfather of Richard Beauchamp, 1St Earl Of Worcester?
4. total=33.15s, path=29.84s, final=3.31s, steps=2, question=Who is Charles Willoughby, 10Th Baron Willoughby Of Parham's paternal grandfather?
5. total=33.01s, path=31.52s, final=1.49s, steps=3, question=Who is James Lyon, 7Th Earl Of Strathmore And Kinghorne's paternal grandfather?
6. total=32.17s, path=28.49s, final=3.68s, steps=3, question=Who is younger, Denise Kandel or Bruce Robinson?
7. total=31.81s, path=31.20s, final=0.62s, steps=3, question=Where did Adolph Of Cleves, Lord Of Ravenstein's mother die?
8. total=31.54s, path=31.02s, final=0.51s, steps=3, question=Were Jimmy Santiago Baca and Duane Armstrong from the same country?
9. total=31.24s, path=30.71s, final=0.52s, steps=3, question=Are Antoine Jean-Baptiste Thomas and Canardo (Rapper) of the same nationality?
10. total=30.81s, path=26.99s, final=3.82s, steps=3, question=Who is the maternal grandmother of Prince Dmitri Alexandrovich Of Russia?

[Notes]
- The original eval summary stores path generation time in the third slot of time[].
- In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.
- avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.
- Predictions are normalized to strip SubQuery/SubAnswer/Final Answer wrappers before answer-quality scoring.
- Low recall often comes from subquery relation drift or overly broad entity resolution before retrieval, not only from the retriever itself.
```
