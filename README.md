# Multi-Hop-QA
Term paper for trends in NLP
MODEL=gemini_pro
python3 -m src.scoring.scoring_v2 \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --gold-csv "src/data_50/mhqa_questions_50.csv" \
  --context-csv "src/data_50/mhqa_context_50.csv" \
  --paras-csv "src/data_50/mhqa_paraphrases_50.csv" \
  --out-json "src/results_50/$MODEL/metrics/${MODEL}_scoring_v2.json" && \
python3 -m src.scoring.scoring_v2_extended \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --gold-csv "src/data_50/mhqa_questions_50.csv" \
  --out-json "src/results_50/$MODEL/metrics/${MODEL}_scoring_v2_extended.json" && \
python3 -m src.scoring.bertscore_scoring_v2 \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --gold-csv "src/data_50/mhqa_questions_50.csv" \
  --model "$MODEL" \
  --outdir "src/results_50/$MODEL/metrics" \
  --bertscore-model "roberta-large" \
  --bertscore-lang "en"

  
  add a good comment and explanation on what this file does . should be professional .must include
   what it does , the functions , how to use it , if inputs needed , outputs to where , if it takes args and defaults ,and small example