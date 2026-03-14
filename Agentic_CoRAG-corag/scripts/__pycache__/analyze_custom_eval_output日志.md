root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# python3 scripts/analyze_custom_eval.py
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
first_step_failure_rate=0.6316
rewrite_after_failure_rate=0.9271
rewrite_salvage_rate=0.3483
relation_drift_count=4
comparison_one_side_missing_rate=0.4722
complex_kinship_failure_rate=0.5294
role_bridge_failure_rate=0.7838
corag_format_issue_total=0
naive_format_issue_total=0

[Category Breakdown]
comparison: count=36, zero_recall_rate=0.5000, avg_corag_recall=0.3042
complex_kinship: count=34, zero_recall_rate=0.5294, avg_corag_recall=0.3676
role_bridge: count=37, zero_recall_rate=0.7838, avg_corag_recall=0.1351
other: count=45, zero_recall_rate=0.5778, avg_corag_recall=0.2778

[CoRAG Recall]
micro=0.2599, macro=0.2694, hits=85, gold=327

[Naive Recall]
micro=0.0550, macro=0.0576, hits=18, gold=327
Recall comparison: corag_better=53, naive_better=5, tie=94

[Top Failure Examples]
1. recall=0.0000, question=Who is the father-in-law of Sisowath Kossamak?, first_subquery=SubQuery: Who is Sisowath Kossamak's spouse?, first_subanswer=SubAnswer: No relevant information found
2. recall=0.0000, question=Where did the composer of film The Straw Hat die?, first_subquery=SubQuery: Who composed the music for the film The Straw Hat?, first_subanswer=SubAnswer: No relevant information found
3. recall=0.0000, question=Who is the child of the director of film An Event?, first_subquery=SubQuery: Who directed the film "An Event"?, first_subanswer=SubAnswer: No relevant information found
4. recall=0.0000, question=Are Christopher Newton (Criminal) and Frances M. Vega of the same nationality?, first_subquery=SubQuery: Who is the nationality of Christopher Newton (Criminal)?, first_subanswer=SubAnswer: No relevant information found
5. recall=0.0000, question=Where did Saw Thanda's husband die?, first_subquery=SubQuery: Who was Saw Thanda's husband?, first_subanswer=SubAnswer: No relevant information found
6. recall=0.0000, question=Did the movies Inside The Room and Crude Set Drama, originate from the same country?, first_subquery=SubQuery: What country is the origin of the movie "Inside The Room"?, first_subanswer=SubAnswer: No relevant information found
7. recall=0.0000, question=What is the date of birth of Henry I Of Ziębice's father?, first_subquery=SubQuery: Who was the father of Henry I Of Ziębice?, first_subanswer=SubAnswer: Nicholas the Small (Mikołaj Mały)
8. recall=0.0000, question=Who is Sibyl Hathaway's child-in-law?, first_subquery=SubQuery: Who is Sibyl Hathaway's spouse?, first_subanswer=SubAnswer: No relevant information found

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
- Retrieval diagnostics now separate first-hop failure, rewrite salvage, relation drift, category-specific failure rates, and result-format anomalies.
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 222222222222222222222222222222222222^C
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# 
root@ubuntu:/home/wangt/corag_dev_lmx/Agentic_CoRAG-corag# python3 scripts/analyze_custom_eval.py
Input eval file: /home/wangt/corag_dev_lmx/Agentic_CoRAG-corag/eval/2wikimultihopqa_hard_small_eval_output_3_14_16_50.json
=== Eval Analysis ===
Samples: 24

[Answer Quality]
EM: 0.0417
Token F1: 0.0580
Contains Match: 0.1250
Yes/No Accuracy: 0.1250 (n=8)
No Relevant Information Rate: 0.5417

[Timing]
path_generation: mean=50.74s, median=56.23s, p90=75.69s, p95=79.53s, max=85.54s
final_generation: mean=1.03s, median=0.00s, p90=2.40s, p95=2.97s, max=3.75s
end_to_end: mean=51.77s, median=57.72s, p90=76.67s, p95=81.60s, max=85.54s

[Reasoning]
avg_steps=2.88, median_steps=3.00, min_steps=2, max_steps=3

[Retrieval Diagnostics]
zero_corag_recall_rate=0.6667
no_info_rate=0.5417
yes_no_error_rate=0.8750
first_step_failure_rate=0.7083
rewrite_after_failure_rate=1.0000
rewrite_salvage_rate=0.2353
relation_drift_count=1
comparison_one_side_missing_rate=0.6250
complex_kinship_failure_rate=1.0000
role_bridge_failure_rate=0.7500
corag_format_issue_total=0
naive_format_issue_total=0

[Category Breakdown]
comparison: count=8, zero_recall_rate=0.2500, avg_corag_recall=0.2750
complex_kinship: count=6, zero_recall_rate=1.0000, avg_corag_recall=0.0000
role_bridge: count=4, zero_recall_rate=0.7500, avg_corag_recall=0.2500
other: count=6, zero_recall_rate=0.8333, avg_corag_recall=0.0833

[CoRAG Recall]
micro=0.1475, macro=0.1542, hits=9, gold=61

[Naive Recall]
micro=0.0164, macro=0.0104, hits=1, gold=61
Recall comparison: corag_better=8, naive_better=1, tie=15

[Top Failure Examples]
1. recall=0.0000, question=Who is the child of the performer of song Me And Bobby Mcgee?, first_subquery=Who performed the song "Me And Bobby Mcgee"?, first_subanswer=No relevant information found
2. recall=0.0000, question=What is the date of birth of Henry I Of Ziębice's father?, first_subquery=Who is the father of Henry I Of Ziębice?, first_subanswer=Nicholas the Small, Duke of Ziębice.
3. recall=0.0000, question=What nationality is the composer of song Make The World Move?, first_subquery=Who composed the song "Make The World Move"?, first_subanswer=No relevant information found
4. recall=0.0000, question=Which award the director of film Edelweiss Pirates (Film) received?, first_subquery=Who directed the film "Edelweiss Pirates"?, first_subanswer=No relevant information found
5. recall=0.0000, question=Where does Gwen Graham's father work at?, first_subquery=Who is Gwen Graham's father?, first_subanswer=No relevant information found
6. recall=0.0000, question=Who is the father-in-law of Sisowath Kossamak?, first_subquery=Who is Sisowath Kossamak's spouse?, first_subanswer=No relevant information found
7. recall=0.0000, question=Who is Godomar Ii's stepmother?, first_subquery=Who was Godomar Ii's mother?, first_subanswer=No relevant information found
8. recall=0.0000, question=Who is Princess Mafalda Of Savoy's maternal grandmother?, first_subquery=Who is Princess Mafalda Of Savoy's mother?, first_subanswer=No relevant information found

[Slowest Samples]
1. total=85.54s, path=85.54s, final=0.00s, steps=3, question=Do both directors of films Friday The 13Th (1916 Film) and Beaumarchais (Film) share the same nationality?
2. total=82.42s, path=80.15s, final=2.27s, steps=3, question=Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?
3. total=76.96s, path=74.95s, final=2.01s, steps=3, question=Where did the composer of film The Straw Hat die?
4. total=76.00s, path=76.00s, final=0.00s, steps=3, question=Who is the child of the performer of song Me And Bobby Mcgee?
5. total=68.68s, path=66.23s, final=2.45s, steps=3, question=Are director of film Raitu Kutumbam and director of film Closet Land both from the same country?
6. total=68.37s, path=66.86s, final=1.50s, steps=3, question=Were both Pietro Salini and Domenico Ravenna, born in the same place?
7. total=68.25s, path=68.25s, final=0.00s, steps=3, question=What nationality is the composer of song Make The World Move?
8. total=63.48s, path=61.78s, final=1.70s, steps=2, question=Are both movies, Naked Tango and Algiers (Film), from the same country?
9. total=63.48s, path=63.48s, final=0.00s, steps=3, question=Which film has the director died first, Showdown At Boot Hill or The Walls Of Hell?
10. total=59.75s, path=57.68s, final=2.07s, steps=3, question=Do both films Once Upon A Time In America and Naked Violence (Film) have the directors that share the same nationality?

[Notes]
- The original eval summary stores path generation time in the third slot of time[].
- In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.
- avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.
- Predictions are normalized to strip SubQuery/SubAnswer/Final Answer wrappers before answer-quality scoring.
- Retrieval diagnostics now separate first-hop failure, rewrite salvage, relation drift, category-specific failure rates, and result-format anomalies.