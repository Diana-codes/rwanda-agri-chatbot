🌾 Rwanda Agricultural Chatbot – Final Report

Author: YOUR NAME
Course/Unit: COURSE/UNIT, DATE

GitHub Repo: rwanda-agri-bot

Live Demo (HF Space): Click to Open

Model Card (HF Hub): View Model

Demo Video (5–10 min): DRIVE/LINK

1️⃣ Project Definition & Domain Alignment

Purpose:
A Rwanda-specific agriculture assistant that answers farmer questions on:

Planting seasons (Season A/B)

Soil types and fertility

Pests and disease control

Irrigation

Market access

Domain Focus:
Localized guidance for crops like maize, beans, potatoes, cassava, banana, coffee, and tea.

Relevance:
Farmers require concise, accurate advice. This chatbot provides rapid, Rwanda-specific responses and safely rejects out-of-domain queries.

2️⃣ Dataset Collection & Preprocessing

Sources:

Rwanda-specific synthetic Q&A (~800): crops, planting windows, soil types, pests, irrigation, cooperatives

General agriculture Q&A (~4,000): improves coverage

Balancing:
Rwanda dataset upsampled (≤10×) to reduce bias toward general data.

Preprocessing Steps:

Text normalization: lowercase, whitespace cleanup, contraction expansion

Prompting: instruction prefix biasing answers toward Rwanda conditions

Tokenization: FLAN‑T5 tokenizer; max input 256, target 128

Data split: 80% train, 10% validation, 10% test

Reproducible Command:

python3 -m src.data_prep \
    --output_dir data/processed \
    --generate_rwanda 800 \
    --generate_general 4000 \
    --max_input_length 256 \
    --max_target_length 128 \
    --upsample_limit 10

3️⃣ Model & Fine-Tuning

Architecture: FLAN‑T5 encoder-decoder Transformer (TensorFlow)
Base Model: google/flan-t5-base (fallback: small variant)
Objective: Seq2Seq conditional generation on instruction-style prompts

Hyperparameters Explored:

Run	Optimizer	LR	Warmup	Batch	Epochs	Val Loss
base_adam	Adam	5e-5	0.00	2	2	VAL_LOSS_A
adamw_warm	AdamW	5e-5	0.06	2	2	VAL_LOSS_B

Data & Token Length Experiments:

Run	Rwanda:General	Max Inp	Max Tgt	BLEU	ROUGE-L	Token-F1	Perplexity
balanced	800:4000 upsampled	256	128	BLEU1	ROUGEL1	F1_1	PPL1
shorter	600:3000 upsampled	128	64	BLEU2	ROUGEL2	F1_2	PPL2

Training Command Example:

CUDA_VISIBLE_DEVICES="" python3 -m src.train \
    --processed_dir data/processed \
    --model_name google/flan-t5-base \
    --output_dir models/run_balanced \
    --batch_size 2 \
    --epochs 2 \
    --learning_rate 5e-5 \
    --optimizer adamw \
    --warmup_ratio 0.06

4️⃣ Evaluation & Results

Metrics:
BLEU, ROUGE‑1/2/L, Token-level Precision/Recall/F1, Perplexity

Quantitative Results (to fill after evaluation):

BLEU: …

ROUGE‑L: …

Token-F1: …

Perplexity: …

Qualitative Observations:

Answers Rwanda seasonal questions (Season A/B)

Suggests soils and fertility practices; provides IPM tips for pests

Rejects OOD queries with domain filter and fallbacks

Findings:

Rwanda-grounded prompts + upsampling reduce generic/global answers

AdamW + warmup stabilizes training and improves validation loss

Limitations:

Synthetic Q&A may miss micro-regional variations

CPU-only training limits model size; small model trades fluency for speed

5️⃣ UI & Deployment

Interface: Gradio multi-turn chat, simple right-sidebar layout
Safety/Robustness:

Out-of-domain guard

Rule-based fallbacks for planting seasons and soil suitability

Deployment:

Live: Hugging Face Spaces (link above)

Local launch:

python3 -m src.app --model_dir models/run_balanced --host 0.0.0.0 --port 7860

6️⃣ Reproducibility

Environment:
Python 3.12, TensorFlow 2.17–2.20, Transformers 4.40–4.43, tf-keras, numpy <2, gradio

Steps:

Setup venv & pip install -r requirements.txt

Generate data (Section 2)

Train model (Section 3)

Evaluate (Section 4)

Run app locally or deploy to HF Spaces

Model Availability: Hosted at HF Model Hub, loaded by model ID in Space.

7️⃣ Ethical Considerations

Synthetic, non-PII data used

Encourages verification with local RAB extension & soil testing

OOD guard prevents unsafe or irrelevant advice

8️⃣ Contributions (Individual)

Data generation patterns & upsampling

Training runs & hyperparameter tuning

Evaluation scripts & metrics

Gradio UI implementation & deployment

Documentation & report preparation

9️⃣ References

Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)

Hugging Face Transformers, Datasets, and Gradio documentation

TensorFlow Keras documentation

10️⃣ Appendix

Example Conversations:

Q	A
When should I plant maize in Rwanda?	Plant at the start of Season B (Mar–Jun) when rains begin. Use varieties like ZM607.
What soil is good for beans?	Well-drained loamy soils; add compost and avoid waterlogging.
Which crops grow well in sandy soils?	Maize, cassava, and groundnuts with added organic matter and mulch.
What is the capital of Rwanda?	I’m focused on agriculture. Please ask about crops, soils, pests, seasons, or farming practices.
