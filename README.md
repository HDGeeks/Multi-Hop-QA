          ┌────────────────────────────┐
          │   Raw predictions (JSONL)  │
          │   per model/run/setting    │
          └──────────────┬─────────────┘
                         │
       ┌─────────────────┼───────────────────┐
       │                 │                   │
       ▼                 ▼                   ▼
┌────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ scoring_v2 │   │ BERTScore v2     │   │ scoring_v2_extended │
│ (Exact/F1) │   │ (Semantic sim.)  │   │ (JSON rollup)       │
└─────┬──────┘   └──────────┬───────┘   └──────────┬──────────┘
      │                     │                      │
      │                     │                      │
      ▼                     ▼                      ▼
  ┌───────────────┐   ┌────────────────┐     ┌─────────────────────────┐
  │ per_run_v2.csv│   │ _per_run_v2.csv│     │ _scoring_v2_extended.json│
  │ (all preds)   │   │ (all preds)    │     │ (overall + per-setting   │
  │ Debug view    │   │ Debug view     │     │ + per-domain JSON)       │
  └─────┬─────────┘   └───────┬────────┘     └─────────────────────────┘
        │                     │
        ▼                     ▼
┌───────────────┐       ┌──────────────────────┐
│ aggregated_   │       │ _aggregated_items_   │
│ items_v2.csv  │       │ v2.csv (per-qid)     │
│ (per qid+run) │       │ + % + drops          │
│ Stability view│       │ Semantic per-item    │
└─────┬─────────┘       └──────────┬───────────┘
      │                            │
      ▼                            ▼
┌───────────────┐         ┌────────────────────┐
│ summary_v2.csv│         │ _by_domain_v2.csv  │
│ (per model×   │         │ Median GOLD/DIST   │
│ setting table)│         │ per domain         │
│ Paper-ready   │         │ Domain insights    │
└───────────────┘         └─────────┬──────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │ _summary_v2.json   │
                          │ Compact medians +  │
                          │ drops vs GOLD      │
                          │ JSON for plots     │
                          └────────────────────┘