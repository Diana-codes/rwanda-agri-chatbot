Rwanda Agricultural Chatbot

Project summary
- Domain-specific chatbot for Rwanda agriculture, using FLAN‑T5 (generative QA) fine-tuned on synthetic Rwanda Q&A plus general agriculture Q&A. Deployed via Gradio with multi-turn chat and OOD guarding.

Setup
1) Python 3.12, create venv and install:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

Data (synthetic, no CSVs required)
Generate/tokenize:
   python3 -m src.data_prep \
     --output_dir data/processed \
     --generate_rwanda 800 --generate_general 4000 \
     --max_input_length 256 --max_target_length 128 --upsample_limit 10

Train (TensorFlow)
Baseline (CPU-friendly):
   CUDA_VISIBLE_DEVICES="" python3 -m src.train \
     --processed_dir data/processed \
     --model_name google/flan-t5-base \
     --output_dir models/run_balanced \
     --batch_size 2 --epochs 2 --learning_rate 5e-5 --optimizer adamw --warmup_ratio 0.06

Evaluate (BLEU, ROUGE, token-F1, Perplexity)
   python3 -m src.evaluate \
     --processed_dir data/processed \
     --model_dir models/run_balanced \
     --num_examples 200 --report_perplexity

Run the app
   python3 -m src.app --model_dir models/run_balanced --host 0.0.0.0 --port 7860

What’s inside (rubric alignment)
- Project definition & domain: `README.md` (this file) + Gradio prompts justify domain and need
- Dataset & preprocessing: `src/data_prep.py` generates Rwanda Q&A (crops, seasons, soils, pests), normalization, tokenization
- Model fine-tuning: `src/train.py` (TF, AdamW, warmup, resume; logs `training_summary.json`)
- Metrics: `src/evaluate.py` reports BLEU, ROUGE, token-F1; optional Perplexity
- UI: `src/app.py` multi-turn chat, OOD guard, rule-based safety fallbacks
- Example conversations: See demo section below

Experiments to report
- Learning rate: 3e-5 vs 5e-5
- Optimizer: Adam vs AdamW (warmup 0.06)
- Batch size: 2 vs 8 (with smaller model)
Record each run’s `models/<run>/training_summary.json` and evaluation outputs into your report tables.

Demo video (5–10 minutes)
- Show setup, data generation, training (short run), evaluation metrics, and UI interactions (in/out of domain, seasonal Q&A). Include findings and limitations.

Notes
- CPU-only runs supported; for speed use smaller model `google/flan-t5-small`.
- Conversation logs are disabled by default; UI is clean and lightweight.
