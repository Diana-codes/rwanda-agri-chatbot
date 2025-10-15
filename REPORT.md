Rwanda Agricultural Chatbot – Report (PDF-ready)

Author: YOUR NAME
Course: DATE / UNIT
Repo: LINK_TO_GITHUB
Demo video (5–10 min): LINK_TO_VIDEO

1. Project definition and domain alignment
- Purpose: Build a Rwanda agriculture chatbot to answer farmer questions on planting seasons, soil, pests, irrigation, and market access.
- Justification: Localized guidance for Seasons A/B, crops and soils common in Rwanda. Many farmers lack quick, consistent advisory.
- Approach: Generative QA with FLAN‑T5, fine-tuned on domain Q&A.

2. Dataset and preprocessing
- Data source: Synthetic Q&A generated programmatically (Rwanda-specific + general agriculture); diverse intents (crops, seasons, soils, pests, irrigation, markets). No personal data.
- Preprocessing: Lowercasing, whitespace normalization, contractions, tokenization with FLAN‑T5 tokenizer, fixed max lengths (inputs/targets), train/val/test split (80/10/10).
- Balancing: Rwanda set upsampled (cap 10x) to reduce bias toward general agri data.
- Example prompts: Instructional prompt grounding to Rwanda (Seasons A/B).

3. Model and training
- Base model: google/flan-t5-base (Transformer, encoder–decoder) implemented in TensorFlow.
- Hyperparameters: batch size (2–16), epochs (1–3), learning rate (3e-5–5e-5), optimizer (Adam vs AdamW), warmup (0–0.06).
- Training infra: CPU-only; resume support with checkpoints; config/metrics saved to models/<run>/training_summary.json.

4. Experiments (tables)
- Table A – Optimization (example):
  | Run | Optimizer | LR   | Warmup | Batch | Epochs | Val Loss |
  |-----|-----------|------|--------|-------|--------|----------|
  | base_adam | Adam | 5e-5 | 0.00 | 2 | 2 | … |
  | adamw_warm | AdamW | 5e-5 | 0.06 | 2 | 2 | … |

- Table B – Data/lengths (example):
  | Run | Rwanda:General | Max Inp | Max Tgt | BLEU | ROUGE-L | Tok-F1 | PPL |
  |-----|----------------|---------|---------|------|---------|--------|-----|
  | balanced | 800:4000 upsampled | 256 | 128 | … | … | … | … |
  | shorter | 600:3000 upsampled | 128 | 64 | … | … | … | … |

5. Evaluation and results
- Metrics: BLEU, ROUGE-1/2/L, token-level Precision/Recall/F1; optional Perplexity. 100–200 test examples.
- Qualitative: Interactive tests for maize/beans planting seasons, soil suitability; OOD prompts are rejected politely.
- Findings: Rwanda-grounded prompts + upsampling improved seasonal accuracy; AdamW+warmup stabilized training.
- Limitations: Synthetic data may miss regional micro-variance; CPU training limits scale.

6. UI and deployment
- Interface: Gradio multi-turn chat; simple domain guard; rule-based fallbacks for critical intents (season/soil) to avoid generic answers.
- Usage: python3 -m src.app --model_dir models/run_balanced --host 0.0.0.0 --port 7860

7. How to reproduce
1) Setup venv and install requirements (README)
2) Data prep with upsampling
3) Train (short run first), then evaluate
4) Launch UI and test

8. Contributions (individual)
- Briefly summarize your work across data, model, evaluation, and UI.

9. References
- FLAN‑T5, Hugging Face Transformers, Datasets, TensorFlow, Gradio docs.

Appendix
- Example conversations (copy a few good Q/A)
- Selected training logs and evaluation outputs

